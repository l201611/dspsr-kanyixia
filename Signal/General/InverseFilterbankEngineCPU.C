//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 - 2025 by Dean Shaff, Andrew Jameson, and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #define _DEBUG 1
#include "debug.h"

#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/TimeSeries.h"
#include "dsp/Scratch.h"
#include "dsp/Apodization.h"
#include "dsp/OptimalFFT.h"
#include "dsp/Observation.h"

#include "FTransform.h"
#include "strutil.h"

#include <fstream>
#include <iostream>
#include <assert.h>
#include <cstring>
#include <complex>
#include <algorithm>

using namespace std;

static const int floats_per_complex = 2;
static const int sizeof_complex = sizeof(float) * floats_per_complex;

#define mod(a,b) (a)%(b)

dsp::InverseFilterbankEngineCPU::InverseFilterbankEngineCPU ()
{
  verbose = Observation::verbose;
  report = false;
}

dsp::InverseFilterbankEngineCPU::~InverseFilterbankEngineCPU ()
{
}

void dsp::InverseFilterbankEngineCPU::setup (dsp::InverseFilterbank* filterbank)
{
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::setup" << endl;

  verbose = filterbank->verbose;
  const TimeSeries* input = filterbank->get_input();
  TimeSeries* output = filterbank->get_output();

  real_to_complex = (input->get_state() == Signal::Nyquist);
  dual_sideband = input->get_dual_sideband();

  pfb_dc_chan = input->get_pfb_dc_chan();
  pfb_all_chan = filterbank->get_pfb_all_chan();
  pfb_nchan = input->get_pfb_nchan();

  input_nchan = input->get_nchan();
  output_nchan = output->get_nchan();

  input_fft_length = filterbank->get_input_fft_length();
  output_fft_length = filterbank->get_output_fft_length();

  input_discard_pos = filterbank->get_input_discard_pos();
  input_discard_neg = filterbank->get_input_discard_neg();
  output_discard_pos = filterbank->get_output_discard_pos();
  output_discard_neg = filterbank->get_output_discard_neg();

  input_discard_total = input_discard_neg + input_discard_pos;
  input_sample_step = input_fft_length - input_discard_total;

  output_discard_total = output_discard_neg + output_discard_pos;
  output_sample_step = output_fft_length - output_discard_total;

  if (filterbank->has_response())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup setting response" << endl;
 
    response = filterbank->get_response();
  }

  if (filterbank->has_temporal_apodization())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup temporal apodization" << endl;

    temporal_apodization = filterbank->get_temporal_apodization();
    if (verbose)
    {
      cerr << "dsp::InverseFilterbankEngineCPU::setup temporal_apodization.get_type() "
           << temporal_apodization->get_type() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::setup temporal_apodization.get_ndim() "
           << temporal_apodization->get_ndim() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::setup temporal_apodization.get_ndat() "
           << temporal_apodization->get_ndat() << endl;
    }
  }

  if (filterbank->has_spectral_apodization())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup spectral apodization" << endl;

    spectral_apodization = filterbank->get_spectral_apodization();
    if (verbose)
    {
      cerr << "dsp::InverseFilterbankEngineCPU::setup spectral_apodization.get_type() "
           << spectral_apodization->get_type() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::setup spectral_apodization.get_ndim() "
           << spectral_apodization->get_ndim() << endl;
      cerr << "dsp::InverseFilterbankEngineCPU::setup spectral_apodization.get_ndat() "
           << spectral_apodization->get_ndat() << endl;
    }
  }

  if (filterbank->get_zero_DM())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup setting zero_DM_response" << endl;

    zero_DM_response = filterbank->get_zero_DM_response();
  }

  OptimalFFT* optimal = 0;
  if (response && response->has_optimal_fft())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup getting OptimalFFT object" << endl;

    optimal = response->get_optimal_fft();
  }

  if (optimal)
    FTransform::set_library( optimal->get_library(input_fft_length) );

  forward = FTransform::Agent::current->get_plan(
      input_fft_length,
      real_to_complex ? FTransform::frc: FTransform::fcc);

  if (optimal)
    FTransform::set_library( optimal->get_library(output_fft_length) );

  backward = FTransform::Agent::current->get_plan (output_fft_length, FTransform::bcc);
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::setup FFT plans created" << endl;
  
  fft_plans_setup = true;

  if (verbose)
  {
    cerr << "dsp::InverseFilterbankEngineCPU::setup"
      << " input_nchan=" << input_nchan
      << " output_nchan=" << output_nchan
      << " pfb_nchan=" << pfb_nchan
      << " input_fft_length=" << input_fft_length
      << " output_fft_length=" << output_fft_length
      << " input_discard_pos=" << input_discard_pos
      << " input_discard_neg=" << input_discard_neg
      << " output_discard_pos=" << output_discard_pos
      << " output_discard_neg=" << output_discard_neg
      << endl;
  }

  // compute oversampling keep/discard region
  input_os_keep = filterbank->get_oversampling_factor().normalize(input_fft_length);
  input_os_discard = input_fft_length - input_os_keep;

  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::setup"
      << " input_os_keep=" << input_os_keep
      << " input_os_discard=" << input_os_discard
      << endl;

  if (input_os_discard % 2 != 0)
    throw Error (InvalidState, "dsp::InverseFilterbankEngineCPU::setup",
		 "input_os_discard=%u must be divisible by two",
		 input_os_discard);

  // input channels are called fine channels
  // by default, there are pfb_nchan fine channels per coarse channel
  fine_channels_per_coarse_channel = pfb_nchan;

  // for now, the input must consist of an integer number of coarse channels 
  if ( input_nchan % fine_channels_per_coarse_channel != 0 )
    throw Error (InvalidState, "dsp::InverseFilterbankEngineCPU::setup",
                 "input_nchan=%u is not divisible by fine_channels_per_coarse_channel=%u",
                 input_nchan, fine_channels_per_coarse_channel);

  coarse_channels = input_nchan / fine_channels_per_coarse_channel;

  assert ( coarse_channels > 0 );
  assert ( input_nchan % output_nchan == 0 );

  // input channels are considered fine channels
  fine_channels_per_output_channel = input_nchan / output_nchan;

  // by default, fine channels spanning a single coarse channel are combined / inverted
  coarse_channels_per_output_channel = 1;

  // by default, all fine channels spanning a single coarse channel are combined / inverted
  output_channels_per_coarse_channel = 1;

  // by default, the number of coarse steps is equal to the number of coarse channels
  spectral_steps = coarse_channels;

  // therefore, the number of output channels per coarse step is 1
  output_channels_per_spectral_step = 1;

  if (fine_channels_per_output_channel > fine_channels_per_coarse_channel)
  {
    // each output channel is composed of fine channels spanning multiple coarse channels

    if (fine_channels_per_output_channel % fine_channels_per_coarse_channel != 0)
      throw Error (InvalidState, "dsp::InverseFilterbankEngineCPU::setup",
                   "combine multiple coarse channels: fine_channels_per_output_channel=%d not divisible by fine_channels_per_coarse_channel=%d",
                    fine_channels_per_output_channel, fine_channels_per_coarse_channel);

    coarse_channels_per_output_channel = fine_channels_per_output_channel / fine_channels_per_coarse_channel;
    spectral_steps = coarse_channels / coarse_channels_per_output_channel;
    assert (spectral_steps == output_nchan);

    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup combining fine channels that span "
           << coarse_channels_per_output_channel << " coarse channels" << endl;

    if (fine_channel_plan.size() == 0)
      build_fine_channel_plan();
  }
  
  if (fine_channels_per_output_channel < fine_channels_per_coarse_channel)
  {
    // each coarse channel is divided into multiple output channels

    if (fine_channels_per_coarse_channel % fine_channels_per_output_channel != 0)
      throw Error (InvalidState, "dsp::InverseFilterbankEngineCPU::setup",
                   "divide coarse channels: fine_channels_per_coarse_channel=%d not divisible by fine_channels_per_output_channel=%d", 
                   fine_channels_per_coarse_channel, fine_channels_per_output_channel);

    output_channels_per_coarse_channel = fine_channels_per_coarse_channel / fine_channels_per_output_channel;
    spectral_steps = output_nchan / output_channels_per_coarse_channel;
    assert (spectral_steps == coarse_channels);

    output_channels_per_spectral_step = output_channels_per_coarse_channel;

    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::setup dividing each coarse channel into "
           << output_channels_per_coarse_channel << " coarse channels" << endl;
  }

  shift_by_half_chan = pfb_dc_chan;

  // setup scratch space
  input_fft_scratch_nfloat = input->get_ndim()*input_fft_length;
  input_time_scratch_nfloat = input_fft_scratch_nfloat;
  output_fft_scratch_nfloat = 2*output_fft_length; // always return complex result

  stitch_scratch_nfloat = 2*output_fft_length * output_channels_per_coarse_channel;

  total_scratch_needed = input_fft_scratch_nfloat +
                         input_time_scratch_nfloat +
                         output_fft_scratch_nfloat +
                         stitch_scratch_nfloat;

  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::setup"
      << " input_fft_scratch_nfloat=" << input_fft_scratch_nfloat
      << " output_fft_scratch_nfloat=" << output_fft_scratch_nfloat
      << endl;

  if (zero_DM_response != nullptr)
    total_scratch_needed += stitch_scratch_nfloat;

  // dsp::Scratch* scratch = new Scratch;
  // input_fft_scratch = scratch->space<float>
  //   (input_time_points + input_fft_points + output_fft_points  + stitch_points);
  // input_time_scratch = input_fft_scratch + input_fft_points;
  // output_fft_scratch = input_time_scratch + input_time_points;
  // stitch_scratch = output_fft_scratch + output_fft_points;
}


void dsp::InverseFilterbankEngineCPU::set_scratch (float * _scratch)
{
  scratch = _scratch;
  input_fft_scratch = scratch;
  input_time_scratch = input_fft_scratch + input_fft_scratch_nfloat;
  output_fft_scratch = input_time_scratch + input_time_scratch_nfloat;
  stitch_scratch = output_fft_scratch + output_fft_scratch_nfloat;
  if (zero_DM_response != nullptr)
    stitch_scratch_zero_DM = stitch_scratch + stitch_scratch_nfloat;
}

void dsp::InverseFilterbankEngineCPU::perform (
  const dsp::TimeSeries* in,
  dsp::TimeSeries* out,
  dsp::TimeSeries* zero_DM_out,
  uint64_t npart,
  uint64_t in_step,
  uint64_t out_step
)
{
  this->input = in;
  this->output = out;
  this->zero_DM_out = zero_DM_out;

  if (verbose)
  {
    if (temporal_apodization)
    {
      cerr << "dsp::InverseFilterbankEngineCPU::perform temporal_apodization"
              " type=" << temporal_apodization->get_type() <<
              " ndim=" << temporal_apodization->get_ndim() <<
              " ndat=" << temporal_apodization->get_ndat() << endl;
    }
    cerr << "dsp::InverseFilterbankEngineCPU::perform output_nchan=" << output_nchan << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::perform"
        << " out dim=(" << out->get_nchan()
        << ", " << out->get_npol()
        << ", " << out->get_ndat() << ")" << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::perform"
        << " in dim=(" << in->get_nchan()
        << ", " << in->get_npol()
        << ", " << in->get_ndat() << ")" << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::perform input_nchan="
        << input_nchan << endl;
  }

  const unsigned n_dim = in->get_ndim();
  const unsigned n_pol = in->get_npol();

  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::perform shift_by_half_chan=" << shift_by_half_chan << endl;

  if (verbose)
  {
    cerr << "dsp::InverseFilterbankEngineCPU::perform coarse channels per output channel: " << coarse_channels_per_output_channel << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::perform fine channels per coarse channel: " << fine_channels_per_coarse_channel << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::perform fine channels per output channel: " << fine_channels_per_output_channel << endl;

    if (shift_by_half_chan)
    {
      if (pfb_all_chan)
        cerr << "dsp::InverseFilterbankEngineCPU::perform "
             << "moving first half channel to end of assembled spectrum" << endl;
      else 
        cerr << "dsp::InverseFilterbankEngineCPU::perform "
             << "moving first half channel to end of assembled spectrum and zeroing" << endl;
    }

    cerr << "dsp::InverseFilterbankEngineCPU::perform writing " << npart << " chunks" << endl;
  }

  for (unsigned ipol = 0; ipol < n_pol; ipol++)
  {
    for (uint64_t ipart = 0; ipart < npart; ipart++)
    {
      uint64_t input_time_offset = ipart * n_dim * input_sample_step;
      uint64_t output_time_offset = ipart * floats_per_complex * output_sample_step;

      for (unsigned spectral_istep = 0; spectral_istep < spectral_steps; spectral_istep++)
      {
        unsigned coarse_chan_offset = spectral_istep * coarse_channels_per_output_channel;
        unsigned fine_channel_offset = coarse_chan_offset * fine_channels_per_coarse_channel;
        unsigned output_channel_offset = spectral_istep * output_channels_per_coarse_channel;
        
        DEBUG("dsp::InverseFilterbankEngineCPU::perform step=" << spectral_istep
          << " coarse_chan_offset=" << coarse_chan_offset
          << " fine_channel_offset=" << fine_channel_offset
          << " output_channel_offset=" << output_channel_offset);

        stitch (stitch_scratch, ipol, fine_channel_offset, input_time_offset, shift_by_half_chan);
        filter (stitch_scratch, ipol, output_channel_offset);
        invert (stitch_scratch, ipol, output_channel_offset, output_time_offset);
      } // end for each spectral step
    } // end for each part (time slice)
  } // end for each polarization

  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::perform finish" << endl;
}

void dsp::InverseFilterbankEngineCPU::build_fine_channel_plan ()
{
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::build_fine_channel_plan" << endl;

  std::vector<unsigned> indeces (input_nchan);

  unsigned half_band = input_nchan / 2;
  unsigned half_coarse = fine_channels_per_coarse_channel / 2;

  // stage 1A: roll by half a coarse channel
  for (unsigned ichan = 0; ichan < input_nchan; ichan++)
  {
    indeces[ichan] = (ichan + input_nchan - half_coarse) % input_nchan;
  }

  // stage 1B: swap the two halves of the band
  for (unsigned ichan = 0; ichan < half_band; ichan++)
  {
    std::swap(indeces[ichan], indeces[ichan+half_band]);
  }

  // stage 2A: roll by half a fine channel (performed by stitch)

  // stage 2B: swap the two halves of each coarse channel
  unsigned ncoarse = input_nchan / fine_channels_per_coarse_channel;
  for (unsigned icoarse=0; icoarse < ncoarse; icoarse++)
  {
    unsigned start_chan = icoarse * fine_channels_per_coarse_channel;
    unsigned end_chan = start_chan + half_coarse;
    for (unsigned ichan=start_chan; ichan < end_chan; ichan++)
    {
      std::swap(indeces[ichan], indeces[ichan+half_coarse]);
    }
  }

  fine_channel_plan.resize (input_nchan);
  for (unsigned ichan = 0; ichan < input_nchan; ichan++)
  {
    // invert the re-ordering
    fine_channel_plan[indeces[ichan]] = ichan;
  }

#if _DEBUG
  std::ofstream out ("channel_indeces.txt");
  for (unsigned ichan = 0; ichan < input_nchan; ichan++)
  {
    out << "ichan=" << ichan << " jchan=" << fine_channel_plan[ichan] << " jchan=" << matlab_fine_channel_plan(ichan) << endl;
  }
#endif
}

unsigned dsp::InverseFilterbankEngineCPU::matlab_fine_channel_plan (unsigned jchan)
{
  unsigned output_channel = floor(jchan / fine_channels_per_output_channel);

  unsigned fine_channel = jchan - output_channel * fine_channels_per_output_channel;

  // shift by half a coarse channel width within output channel
  fine_channel = mod ((fine_channel + fine_channels_per_coarse_channel/2), fine_channels_per_output_channel);

  // fprintf ('fine chan per coarse chan = %d\n',fine_channels_per_coarse_channel);

  // re-order input channels in DSB monotonically
  unsigned coarse_channel = floor(fine_channel / fine_channels_per_coarse_channel);
  fine_channel = fine_channel - coarse_channel*fine_channels_per_coarse_channel;

  // fprintf ('chan=%d coarse=%d fine=%d \n', chan, coarse_channel, fine_channel);

  unsigned combine = coarse_channels_per_output_channel;

  // swap halves of the band within the output channel
  coarse_channel = mod ((coarse_channel + combine/2), combine);
  coarse_channel = output_channel * combine + coarse_channel;

  // swap halves of the band within the coarse channel
  fine_channel = mod ((fine_channel + fine_channels_per_coarse_channel/2), fine_channels_per_coarse_channel);
  return coarse_channel * fine_channels_per_coarse_channel + fine_channel;
}

void dsp::InverseFilterbankEngineCPU::stitch (
  float* dest,
  unsigned ipol,
  unsigned fine_channel_offset, 
  uint64_t input_offset_nfloat,
  bool shift_by_half_chan)
{
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::stitch dest=" << dest << " ipol=" << ipol
         << " chan_offset=" << fine_channel_offset << " input time_offset=" << input_offset_nfloat << endl;

  const unsigned n_dim = input->get_ndim();

  for (unsigned ichan_coarse = 0; ichan_coarse < coarse_channels_per_output_channel; ichan_coarse++)
  {
    unsigned fine_channel_start = fine_channel_offset + ichan_coarse * fine_channels_per_coarse_channel;
    uint64_t scratch_offset = ichan_coarse * n_dim * input_os_keep * fine_channels_per_coarse_channel;

    DEBUG("dsp::InverseFilterbankEngineCPU::stitch ichan_coarse=" << ichan_coarse 
          << " fine_channel_start=" << fine_channel_start << " scratch_offset=" << scratch_offset);

    stitch_one (dest+scratch_offset, ipol, fine_channel_start, input_offset_nfloat, shift_by_half_chan);
  }

  // cerr << "last index=" << coarse_channels_per_output_channel * n_dim * input_os_keep * fine_channels_per_coarse_channel << endl;
}

void dsp::InverseFilterbankEngineCPU::stitch_one (
  float* dest, 
  unsigned ipol, 
  unsigned chan_offset,
  uint64_t input_offset_nfloat, 
  bool shift_by_half_chan)
{
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::stitch_one dest=" << dest << " ipol=" << ipol
      << " chan_offset=" << chan_offset << " input_offset_nfloat=" << input_offset_nfloat 
      << " input_os_keep=" << input_os_keep<< " dsb=" << dual_sideband << endl;

  const unsigned n_dim = input->get_ndim();
  const unsigned input_os_keep_2 = input_os_keep / 2;
  const unsigned copy_bytes = input_os_keep_2 * sizeof_complex;

  DEBUG("dsp::InverseFilterbankEngineCPU::stitch memcpy bytes=" << copy_bytes);

  unsigned stitched_offset_pos = 0;
  unsigned stitched_offset_neg = 0;

  unsigned max_offset = 0;

  unsigned nchan = fine_channels_per_coarse_channel;

  static unsigned execution_completed_count = 0;

  for (unsigned input_ichan = 0; input_ichan < nchan; input_ichan++)
  {
    unsigned input_jchan = input_ichan + chan_offset;

    if (coarse_channels_per_output_channel > 1)
      input_jchan = fine_channel_plan[input_jchan];

    DEBUG("dsp::InverseFilterbankEngineCPU::stitch ichan=" << input_ichan << " jchan=" << input_jchan);

    const float* time_dom_ptr = input->get_datptr (input_jchan, ipol) + input_offset_nfloat;

    if (temporal_apodization)
    {
      DEBUG("dsp::InverseFilterbankEngineCPU::stitch perform temporal apodization");

      memcpy (input_time_scratch, time_dom_ptr, input_fft_scratch_nfloat * sizeof(float));
      temporal_apodization->operate (input_time_scratch);
      time_dom_ptr = input_time_scratch;

      if (report)
        reporter.emit("temporal_apodization", input_time_scratch, 1, 1, input_fft_length, 2);
    }

#define CONJUGATE 0
#if CONJUGATE
    memcpy (input_time_scratch, time_dom_ptr, input_fft_scratch_nfloat * sizeof(float));
    time_dom_ptr = input_time_scratch;
    cerr << " conjugate " << endl;
    for (unsigned i=0; i < input_fft_length/2; i++)
      input_time_scratch[i*2+1] *= -1.0;
#endif

#define DUMP_FFT_INPUT 0
#if DUMP_FFT_INPUT

    if (execution_completed_count == 0)
    {
      string filename = stringprintf("fft_input_%04d.dat", input_ichan);

      FILE* fptr = fopen (filename.c_str(), "w");
      for (unsigned i=0; i<input_fft_length; i++)
        fprintf (fptr, "%i %f %f\n", i, time_dom_ptr[i*2], time_dom_ptr[i*2+1]);
      fclose (fptr);
    }

#endif

    DEBUG("dsp::InverseFilterbankEngineCPU::stitch perform forward FFT");

    // f?c1d(number_of_points, dest_ptr, src_ptr)
    if (real_to_complex)
      forward->frc1d(input_fft_length, input_fft_scratch, time_dom_ptr);
    else
      forward->fcc1d(input_fft_length, input_fft_scratch, time_dom_ptr);

    // discard oversampled regions and do circular shift
    if (report)
      reporter.emit("fft", input_fft_scratch, 1, 1, input_fft_length, 2);

    if (shift_by_half_chan)
    {
      DEBUG("dsp::InverseFilterbankEngineCPU::stitch shift by half channel");

      if (input_ichan == 0)
      {
        stitched_offset_neg = n_dim*(input_os_keep*fine_channels_per_coarse_channel - input_os_keep_2);
        stitched_offset_pos = 0;
      }
      else
      {
        stitched_offset_neg = n_dim*(input_os_keep*input_ichan - input_os_keep_2);
        stitched_offset_pos = stitched_offset_neg + n_dim*input_os_keep_2;
      }
    }
    else
    {
      stitched_offset_neg = n_dim*input_os_keep*input_ichan;
      stitched_offset_pos = stitched_offset_neg + n_dim*input_os_keep_2;
    }
              
    const float* input_pos_scratch = input_fft_scratch;
    const float* input_neg_scratch = input_fft_scratch + n_dim*(input_fft_length - input_os_keep_2);

    if (!dual_sideband)
    {
      DEBUG("dsp::InverseFilterbankEngineCPU::stitch not dual sideband");
      auto nskip_2 = (input_fft_length - input_os_keep)/2;
      input_neg_scratch = input_fft_scratch + n_dim*nskip_2;
      input_pos_scratch = input_neg_scratch + n_dim*input_os_keep_2;
    }

    DEBUG("dsp::InverseFilterbankEngineCPU::stitch memcpy stitched_offset_neg=" << stitched_offset_neg);
    memcpy(dest + stitched_offset_neg, input_neg_scratch, copy_bytes);

    max_offset = std::max(max_offset, stitched_offset_neg + input_os_keep_2 * floats_per_complex);

    DEBUG("dsp::InverseFilterbankEngineCPU::stitch memcpy stitched_offset_pos=" << stitched_offset_pos);
    memcpy(dest + stitched_offset_pos, input_pos_scratch, copy_bytes);

    max_offset = std::max(max_offset, stitched_offset_pos + input_os_keep_2 * floats_per_complex);

  } // end of for input_nchan

  execution_completed_count ++;
  // cerr << "dsp::InverseFilterbankEngineCPU::stitch_one max_offset=" << max_offset << endl;
}


/* called once per spectral step */
void dsp::InverseFilterbankEngineCPU::filter (float* spectrum, unsigned ipol, unsigned chan_offset)
{
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::filter input ptr=" << spectrum << " ipol=" << ipol
      << " chan_offset=" << chan_offset << endl;

  const unsigned n_dim = input->get_ndim();
  const unsigned nchan_out = output_channels_per_spectral_step;
  const unsigned input_os_keep_2 = input_os_keep / 2;

  // If we have the zeroth PFB channel and we don't have all the
  // PFB channels, then we zero the last half channel.
  if (! pfb_all_chan && pfb_dc_chan)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::filter zeroing last half channel" << endl;

    unsigned offset = n_dim*(nchan_out*output_fft_length - input_os_keep_2);
    for (unsigned i=0; i<n_dim*input_os_keep_2; i++)
      spectrum[offset + i] = 0.0;
  }

  if (zero_DM_response != nullptr)
  {
    // copy data from spectrum into stitch_scratch_zero_DM
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::filter applying zero_DM_response" << endl;

    memcpy (stitch_scratch_zero_DM, spectrum, nchan_out*output_fft_length*sizeof_complex);
    zero_DM_response->operate(stitch_scratch_zero_DM, ipol, chan_offset, nchan_out);
  }
  else if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::filter NOT applying zero_DM_response" << endl;

  if (response != nullptr)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::filter applying response"
        " chan_offset=" << chan_offset <<
        " nchan_out=" << nchan_out << endl;

    response->operate(spectrum, ipol, chan_offset, nchan_out);
  }
  else if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::filter NOT applying response" << endl;

  if (report)
    reporter.emit("response_stitch", spectrum, 1, 1, output_fft_length*nchan_out, 2);

  if (spectral_apodization != nullptr)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbankEngineCPU::filter spectral apodization" << endl;

    for (unsigned ichan_out=0; ichan_out<nchan_out; ichan_out++)
    {
      spectral_apodization -> operate (spectrum + ichan_out*output_fft_length*n_dim);
    }
  }

  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::filter" << endl;

#define DUMP_FIRST_SPECTRUM 0
#if DUMP_FIRST_SPECTRUM

  static unsigned spectrum_count = 0;
  constexpr unsigned dump_on = 0;

  if (spectrum_count == dump_on)
  {
    FILE* fptr = fopen ("N_spectrum.dat", "w");
    for (unsigned i=0; i<output_fft_length; i++)
    {
      auto re = spectrum[i*2];
      auto im = spectrum[i*2+1];
      auto asq = re*re + im*im;
      float dph = 0.0;
      if (i > 0)
      {
        auto re0 = spectrum[(i-1)*2];
        auto im0 = spectrum[(i-1)*2+1];
        auto cre = re*re0 + im*im0;
        auto cim = re*im0 - im*re0;
        dph = atan2(cim,cre);
      }
      fprintf (fptr, "%i %f %f %f %f\n", i, re, im, asq, dph);
    }

    fclose (fptr);
    cerr << "dsp::InverseFilterbankEngineCPU::filter exit after dumping spectrum number " << dump_on << endl;
    exit(-1);
  }

  spectrum_count ++;

#endif

}

/*! called once per spectral step */
void dsp::InverseFilterbankEngineCPU::invert (float* spectrum, unsigned ipol, unsigned chan_offset, uint64_t output_offset_nfloat)
{
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::invert input ptr=" << spectrum << " ipol=" << ipol
      << " chan_offset=" << chan_offset << " output time_offset=" << output_offset_nfloat 
      << " output_fft_length=" << output_fft_length 
      << " output_discard_pos=" << output_discard_pos 
      << " output_sample_step=" << output_sample_step << endl;

  const unsigned n_dim = input->get_ndim();
  const unsigned nchan_out = output_channels_per_spectral_step;

  for (unsigned ichan_out = 0; ichan_out < nchan_out; ichan_out++)
  {
    // cerr << "dsp::InverseFilterbankEngineCPU::invert before inverse fft length=" << output_fft_length << endl;

    backward->bcc1d(output_fft_length, output_fft_scratch, spectrum + ichan_out*output_fft_length*n_dim);
    if (report)
      reporter.emit("ifft", output_fft_scratch, 1, 1, output_fft_length, 2);

#define DUMP_FIRST_TIME 0
#if DUMP_FIRST_TIME

  static unsigned time_count = 0;
  constexpr unsigned dump_on = 0;

  if (time_count == dump_on)
  {
    FILE* fptr = fopen ("N_timeseries.dat", "w");
    for (unsigned i=0; i<output_fft_length; i++)
    {
      auto re = output_fft_scratch[i*2];
      auto im = output_fft_scratch[i*2+1];
      auto asq = re*re + im*im;
      float dph = atan2(im,re);
      fprintf (fptr, "%i %f %f %f %f\n", i, re, im, asq, dph);
    }

    fclose (fptr);
    cerr << "dsp::InverseFilterbankEngineCPU::filter exit after dumping timeseries number " << dump_on << endl;
    exit(-1);
  }

  time_count ++;

#endif

    // Output is in FPT order.
    void* src_ptr = output_fft_scratch + output_discard_pos*floats_per_complex;
    void* dest_ptr = output->get_datptr(ichan_out + chan_offset, ipol) + output_offset_nfloat;
    memcpy(dest_ptr, src_ptr, output_sample_step*sizeof_complex);

    if (zero_DM_response != nullptr)
    {
      backward->bcc1d(output_fft_length, output_fft_scratch, stitch_scratch_zero_DM + ichan_out*output_fft_length*n_dim);
      src_ptr = output_fft_scratch + output_discard_pos*floats_per_complex;
      dest_ptr = zero_DM_out->get_datptr(ichan_out + chan_offset, ipol) + output_offset_nfloat;
      
      memcpy(dest_ptr, src_ptr, output_sample_step*sizeof_complex);
    }
  } // for each output channel
}

void dsp::InverseFilterbankEngineCPU::perform (
  const dsp::TimeSeries* in,
  dsp::TimeSeries* out,
  uint64_t npart,
  uint64_t in_step,
  uint64_t out_step
)
{
  perform(in, out, nullptr, npart, in_step, out_step);
}

void dsp::InverseFilterbankEngineCPU::finish ()
{
  if (verbose)
    cerr << "dsp::InverseFilterbankEngineCPU::finish" << endl;
}
