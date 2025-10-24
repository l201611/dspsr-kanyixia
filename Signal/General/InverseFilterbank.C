/***************************************************************************
 *
 *   Copyright (C) 2018-2025 by Dean Shaff, Andrew Jameson, and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngine.h"

#include "dsp/FIRFilter.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Response.h"
#include "dsp/Dedispersion.h"
#include "dsp/Apodization.h"
#include "dsp/InputBuffering.h"
#include "dsp/Scratch.h"
#include "dsp/OptimalFFT.h"
#include "Error.h"

#include <vector>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>

using namespace std;

#define _DEBUG 1

dsp::InverseFilterbank::InverseFilterbank (const char* name, Behaviour behaviour)
  : Convolution (name, behaviour)
{
  set_buffering_policy (new InputBuffering (this));
}

void dsp::InverseFilterbank::set_pfb_all_chan (bool _pfb_all_chan)
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::set_pfb_all_chan " << _pfb_all_chan << endl;
  pfb_all_chan = _pfb_all_chan;
}

void dsp::InverseFilterbank::set_input (const dsp::TimeSeries* input)
{
  if (verbose)
  {
    cerr << "dsp::InverseFilterbank::set_input input=" << input << endl;
    cerr << "dsp::InverseFilterbank::set_input oversampling_factor="
         << input->get_oversampling_factor() << endl;
  }
  dsp::Convolution::set_input(input);
  oversampling_factor = input->get_oversampling_factor();

  if (input_fft_length)
    set_input_fft_length(input_fft_length);
}

void dsp::InverseFilterbank::set_input_fft_length (unsigned nfft, const dsp::Observation* info)
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::set_input_fft_length nfft=" << nfft << " Observation*=" << (void*) info << endl;

  input_fft_length = nfft;

  if (!info && !input)
    return;

  if (!info)
    info = input;

  unsigned in_nchan = info->get_nchan();
  unsigned out_nchan = get_output_nchan();
  Rational os = info->get_oversampling_factor();

  freq_res = input_to_output(nfft, in_nchan, out_nchan, os);
  if (Operation::verbose)
    cerr << "dsp::InverseFilterbank::set_input_fft_length os=" << os << " freq_res=" << freq_res << endl;

  if (response)
    response->set_frequency_resolution(freq_res);

  if (zero_DM_response)
    response->set_frequency_resolution(freq_res);
}

//! Set the frequency response function
void dsp::InverseFilterbank::set_response (Response* response)
{
  Convolution::set_response(response);

  if (freq_res)
    response->set_frequency_resolution(freq_res);
}

void dsp::InverseFilterbank::set_zero_DM_response (Response* zdm_response)
{
  Convolution::set_zero_DM_response(zdm_response);

  if (freq_res)
    zdm_response->set_frequency_resolution(freq_res);
}

void dsp::InverseFilterbank::prepare () try
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare" << endl;

  make_preparations ();
  prepared = true;
}
catch (Error& error)
{
  throw error += "dsp::InverseFilterbank::prepare";
}

void dsp::InverseFilterbank::reserve ()
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::reserve" << endl;

  resize_output (true);
}

void dsp::InverseFilterbank::set_engine (Engine* _engine)
{
  engine = _engine;
}

uint64_t dsp::output_to_input (
  uint64_t n,
  unsigned _input_n,
  unsigned _output_n,
  const Rational& osf
)
{
  imaxdiv_t result = imaxdiv (n * _output_n * osf.get_numerator(), _input_n * osf.get_denominator());
  if (result.rem != 0)
    throw Error (InvalidParam, "dsp::output_to_input",
        "n*(Nout/Nin)*os is not an integer n=%u N_in=%u N_out=%u os=%u/%u (%u/%u) rem=%u",
        n, _input_n, _output_n, osf.get_numerator(), osf.get_denominator(),
        n * _output_n * osf.get_numerator(), _input_n * osf.get_denominator(), result.rem);

  return result.quot;
}

uint64_t dsp::input_to_output (
  uint64_t n,
  unsigned _input_n,
  unsigned _output_n,
  const Rational& osf
)
{
  imaxdiv_t result = imaxdiv (n * _input_n * osf.get_denominator(), _output_n * osf.get_numerator());
  if (result.rem != 0)
    throw Error (InvalidParam, "dsp::input_to_output",
        "n*(Nin/Nout)/os is not an integer n=%u N_in=%u N_out=%u os=%u/%u",
        n, _input_n, _output_n, osf.get_numerator(), osf.get_denominator());

  return result.quot;
}

void dsp::InverseFilterbank::transformation () try
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::transformation input"
      " ndat=" << input->get_ndat() << " nchan=" << input->get_nchan() << endl;

  if (!prepared)
    prepare ();

  resize_output ();

  if (has_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::transformation next_start=" << input_sample_step * npart << endl;

    get_buffering_policy()->set_next_start (input_sample_step * npart);
  }

  int64_t input_sample = input->get_input_sample();
  int64_t output_sample = input_to_output ( input_sample, 
                                            input_nchan, output_nchan, 
                                            input->get_oversampling_factor() );

  if (verbose)
    cerr << "dsp::InverseFilterbank::transformation: setting input sample to "
      << output_sample << endl;

  output->set_input_sample (output_sample);
  if (zero_DM)
    get_zero_DM_output()->set_input_sample(output_sample);

  if (verbose)
    cerr << "dsp::InverseFilterbank::transformation after prepare output"
            " ndat=" << output->get_ndat() <<
            " input_sample=" << output->get_input_sample() << endl;

  // WvS AT3-119 - end of not understanding input_sample = ndat

  if (!npart)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::transformation empty result" << endl;
    return;
  }

  filterbank ();
}
catch (Error& error)
{
  throw error += "dsp::InverseFilterbank::transformation";
}

void dsp::InverseFilterbank::filterbank()
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::filterbank" << endl;

  // number of floats to step between input to filterbank
  const uint64_t in_step = input_sample_step * input->get_ndim();
  const uint64_t out_step = output_sample_step * output->get_ndim();

  engine->perform (input, output, zero_DM_output, npart, in_step, out_step);

  if (Operation::record_time)
    engine->finish ();
}

void dsp::InverseFilterbank::make_preparations () try
{
  Rational osf = oversampling_factor = input->get_oversampling_factor();

  unsigned pfb_nchan = input->get_pfb_nchan ();

  if (verbose)
    cerr << "dsp::InverseFilterbank::make_preparations"
      << " oversampling_factor=" << get_oversampling_factor()
      << " input oversampling_factor=" << input->get_oversampling_factor()
      << " pfb_dc_chan=" << input->get_pfb_dc_chan()
      << " pfb_nchan=" << input->get_pfb_nchan()
      << " pfb_all_chan=" << get_pfb_all_chan()
      << endl;
 
  if (pfb_nchan < 2)
    throw Error (InvalidState, "dsp::InverseFilterbank::make_preparations",
                 "input with pfb_nchan=%d", pfb_nchan);
 
  bool real_to_complex = (input->get_state() == Signal::Nyquist);

  /* number of input time samples per complex-valued output time sample
     independent of oversampling */
  
  unsigned nsamp_in_per_nsamp_out = real_to_complex ? 2 : 1;

  // setup the dedispersion discard region for the forward and backward FFTs
  input_nchan = input->get_nchan();

  if (verbose)
    cerr << "dsp::InverseFilterbank::make_preparations has_response=" << has_response() << endl;

  if (has_response())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::make_preparations response ptr=" << (void*) response << endl;

    response->set_downsampling(true);
    response->match(input, output_nchan);

    if (zero_DM && has_zero_DM_response())
      zero_DM_response->match(input, output_nchan);

    output_discard_pos = response->get_impulse_pos();
    output_discard_neg = response->get_impulse_neg();
    freq_res = response->get_ndat();
    
    output_fft_length = freq_res; // WvS AT3-119 freq_res is redundant?

    if (verbose)
      cerr << "dsp::InverseFilterbank::make_preparations have response"
              " output_discard_pos=" << output_discard_pos <<
              " output_discard_neg=" << output_discard_neg <<
              " output_fft_length=" << output_fft_length << endl;

    input_discard_pos = output_to_input (output_discard_pos,
					 input_nchan, output_nchan, osf);
    input_discard_neg = output_to_input (output_discard_neg,
					 input_nchan, output_nchan, osf);
    input_fft_length  = output_to_input (output_fft_length,
					 input_nchan, output_nchan, osf);

    if (verbose)
      cerr << "dsp::InverseFilterbank::make_preparations have response"
              " freq_res=" << freq_res << " discard pos=" << input_discard_pos << endl;

    input_fft_length  *= nsamp_in_per_nsamp_out;
    input_discard_pos *= nsamp_in_per_nsamp_out;
    input_discard_neg *= nsamp_in_per_nsamp_out;

    input_discard_total = input_discard_neg + input_discard_pos;
    input_sample_step = input_fft_length - input_discard_total;

    output_discard_total = output_discard_neg + output_discard_pos;
    output_sample_step = output_fft_length - output_discard_total;

  }
  else
  {
#define AJTESTING
    // we want a 128pt forward FFT in each channel
    unsigned min_fft_length = 1024;
    freq_res = get_oversampling_factor().normalize(min_fft_length * (input_nchan / output_nchan));
    input_fft_length = output_to_input(freq_res, input_nchan, output_nchan, osf);
    output_fft_length = input_nchan*get_oversampling_factor().normalize(input_fft_length) / output_nchan;
    input_fft_length *= nsamp_in_per_nsamp_out;

#ifdef AJTESTING

    cerr << "freq_res=" << freq_res << " discard pos=" << input_discard_pos << endl;

    input_discard_pos = input_discard_neg = input_fft_length / 4;
    input_discard_total = input_discard_neg + input_discard_pos;
    input_sample_step = input_fft_length - input_discard_total;

    output_discard_pos = output_discard_neg = output_fft_length / 4;
    output_discard_total = output_discard_neg + output_discard_pos;
    output_sample_step = output_fft_length - output_discard_total;
#else
    output_sample_step = output_fft_length;
    input_sample_step = input_fft_length;
#endif

    if (verbose)
    {
      cerr << "dsp::InverseFilterbank::make_preparations input_nchan=" << input_nchan 
                << " input_fft_length=" << input_fft_length << " OSfactor.normalize(" << input_fft_length << ")=" << get_oversampling_factor().normalize(input_fft_length) << " output_nchan=" << output_nchan << endl;
      std::cerr << "nsamp_in_per_nsamp_out=" << nsamp_in_per_nsamp_out << endl;
    }
  }

  if (verbose)
  {
    cerr << "dsp::InverseFilterbank::make_preparations input_nchan/output_nchan=" 
      << input_nchan/output_nchan << endl;
    cerr << "dsp::InverseFilterbank::make_preparations output_fft_length="
      << output_fft_length << endl;
    cerr << "dsp::InverseFilterbank::make_preparations input_fft_length="
      << input_fft_length << endl;
    cerr << "dsp::InverseFilterbank::make_preparations output_sample_step="
      << output_sample_step << endl;
    cerr << "dsp::InverseFilterbank::make_preparations input_sample_step="
      << input_sample_step << endl;
    cerr << "dsp::InverseFilterbank::make_preparations"
      << " input_discard_neg=" << input_discard_neg
      << " input_discard_pos=" << input_discard_pos
      << endl;
    cerr << "dsp::InverseFilterbank::make_preparations"
      << " output_discard_neg=" << output_discard_neg
      << " output_discard_pos=" << output_discard_pos
      << endl;

    cerr << "dsp::InverseFilterbank::make_preparations freq_res="
      << freq_res << endl;
  }

  assert (input_nchan % output_nchan == 0);

  if (has_temporal_apodization())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::make_preparations temporal apodization" << endl;

    dsp::Apodization* fft_window = get_temporal_apodization();
    fft_window->set_size (input_fft_length);
    fft_window->set_analytic (true);
    fft_window->set_transition (input_discard_pos);
    fft_window->build ();
    fft_window->dump ("temporal_apodization.dat");
  }

  if (spectral_apodization)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::make_preparations spectral apodization" << endl;

    /* 
       frequency order in output channels depends on mode:
       - true if stitching multiple coarse channels together using fine channel plan
       - false if stitching only fine channels that span a single coarse channel
    */
    bool dual_sideband_channels = input_nchan / output_nchan >= pfb_nchan;

    if (verbose)
      cerr << "dsp::InverseFilterbank::make_preparations prepare_spectral_apodization with dual sideband output = " << dual_sideband_channels << endl;

    prepare_spectral_apodization ( output_fft_length, dual_sideband_channels );
  }

  if (verbose)
  {
    cerr << "dsp::InverseFilterbank::make_preparations done optimizing fft lengths and discard regions" << endl;
    cerr << "dsp::InverseFilterbank::make_preparations input_fft_length="
         << input_fft_length << " output_fft_length="
         << output_fft_length << " input_discard_neg/pos="
         << input_discard_neg << "/" << input_discard_pos
         << " output_discard_neg/pos="
         << output_discard_neg << "/" << output_discard_pos
         << endl;
  }

  if (has_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::make_preparations reserve=" << input_fft_length << endl;
    get_buffering_policy()->set_maximum_samples (input_fft_length);
  }

  scalefac = 1.0;
  if (FTransform::get_norm() == FTransform::unnormalized)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::make_preparations unnormalized FFT" << endl;

    scalefac = pow(double(output_fft_length), 2);
  }
  else if (verbose)
    cerr << "dsp::InverseFilterbank::make_preparations normalized FFT" << endl;

  scalefac *= pow(oversampling_factor.doubleValue(), 2);

  if (verbose)
    cerr << "dsp::InverseFilterbank::make_preparations"
         << " scalefac=" << scalefac << endl;
 
  prepare_output ();

  if (verbose)
    cerr << "dsp::InverseFilterbank::make_preparations "
         << "calling engine setup" << endl;

  engine->setup (this);

  if (verbose)
    cerr << "dsp::InverseFilterbank::make_preparations allocating "
	 << engine->get_total_scratch_needed()
	 << " floats for engine scratch space" << endl;

  float* scratch_space = scratch->space<float>(engine->get_total_scratch_needed());
  engine->set_scratch(scratch_space);
}
catch (Error& error)
{
  throw error += "dsp::InverseFilterbank::make_preparations";
}

void dsp::InverseFilterbank::prepare_output (uint64_t ndat, bool set_ndat)
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output ndat=" << ndat
         << " set_ndat=" << set_ndat 
         << " input->ndat=" << get_input()->get_ndat() << endl;

  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output set ndat=" << ndat << endl;

    output->set_npol( input->get_npol() );
    output->set_nchan( output_nchan );
    output->set_ndim( 2 );
    output->set_state( Signal::Analytic);
    output->resize( ndat );
  }

  WeightedTimeSeries* weighted_output;
  weighted_output = dynamic_cast<WeightedTimeSeries*> (output.get());

  /* the problem: copy_configuration copies the weights array, which
     results in a call to resize_weights, which sets some offsets
     according to the reserve (for later prepend).  However, the
     offset is computed based on values that are about to be changed.
     This kludge allows the offsets to reflect the correct values
     that will be set later */

  unsigned tres_ratio = 1;

  if (weighted_output)
    weighted_output->set_reserve_kludge_factor (tres_ratio);

  output->copy_configuration(get_input());

  output->set_nchan(output_nchan);
  output->set_ndim(2);
  output->set_state(Signal::Analytic);

  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output"
         << " output ndim=" << output->get_ndim()
         << " output npol=" << output->get_npol()
         << " output nchan=" << output->get_nchan()
         << " output ndat=" << output->get_ndat()
         << endl;

  custom_prepare ();

  if (weighted_output)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output using weighted_output" << endl;
    weighted_output->set_reserve_kludge_factor (1);
    weighted_output->convolve_weights (input_fft_length, input_sample_step);
    weighted_output->scrunch_weights (tres_ratio);
  }

  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output reset ndat=" << ndat << endl;
    output->resize (ndat);
  }
  else
  {
    /* WvS AT3-119 the following line would almost never be correct.
       It ignores the output_ndat calculation of resize_output */
    ndat = input->get_ndat() / tres_ratio;

    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output scrunch ndat=" << ndat << endl;
    output->resize (ndat);
  }

  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output output ndat=" << output->get_ndat() << endl;

  output->rescale (scalefac);

  if (verbose) 
    cerr << "dsp::InverseFilterbank::prepare_output scale=" << output->get_scale() << endl;

  /*
   * output data will have new sampling rate
   * NOTE: that nsamp_fft already contains the extra factor of two required
   * when the input TimeSeries is Signal::Nyquist (real) sampled
   */

  double ratechange = static_cast<double>(input_nchan) / static_cast<double>(output_nchan);
  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output"
         << " ratechange=" << ratechange
         << " input_nchan=" << static_cast<double>(input_nchan)
         << " output_nchan=" << static_cast<double>(output_nchan)
         << endl;
 
  ratechange /= input->get_oversampling_factor().doubleValue();
	
  output->set_rate (input->get_rate() * ratechange);

  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output"
         << " ratechange=" << ratechange
         << " input rate=" << input->get_rate()
         << " output rate=" << output->get_rate()
         << endl;

  if (freq_res == 1)
    output->set_dual_sideband (true);

  /*
   * if freq_res is even, then each sub-band will be centred on a frequency
   * that lies on a spectral bin *edge* - not the centre of the spectral bin
   */

  output->set_dc_centred (freq_res%2);

  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output"
      << " output->get_dual_sideband()=" << output->get_dual_sideband()
      << " output->get_dc_centred()=" << output->get_dc_centred()
      << endl;

  // increment the start time by the number of samples dropped from the fft

  //cerr << "FILTERBANK OFFSET START TIME=" << output_discard_pos << endl;

  output->change_start_time (output_discard_pos);

  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output start time += "
         << output_discard_pos << " samps -> " << output->get_start_time() << endl;

  if (zero_DM)
  {
    get_zero_DM_output()->copy_configuration(output);
    get_zero_DM_output()->resize(output->get_ndat());
    get_zero_DM_output()->change_start_time(output_discard_pos);
  }
}

void dsp::InverseFilterbank::resize_output (bool reserve_extra)
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::resize_output"
         << " reserve_extra=" << reserve_extra << endl;

  const uint64_t ndat = input->get_ndat();

  // number of big FFTs (not including, but still considering, extra FFTs
  // required to achieve desired time resolution) that can fit into data
  npart = 0;

  if (input_sample_step == 0)
    throw Error (InvalidState, "dsp::InverseFilterbank::resize_output",
                 "input_sample_step == 0 ... not properly prepared");

  if (ndat > (unsigned) input_discard_total)
    npart = (ndat-input_discard_total)/input_sample_step;

  /* On some iterations, ndat could be large enough to fit an extra part.
     Note that the reserve_extra argument is set to true only when this
     method is called by InverseFilterbank::reserve */
  if (reserve_extra && has_buffering_policy())
    npart += 2;

  uint64_t output_ndat = npart * output_sample_step;

  if (verbose)
  {
    cerr << "dsp::InverseFilterbank::resize_output input ndat=" << ndat
         << " overlap=" << input_discard_total << " step=" << input_sample_step
         << " reserve=" << reserve_extra << " output_sample_step=" << output_sample_step
         << " npart=" << npart << " output ndat=" << output_ndat << endl;
  }

  // prepare the output TimeSeries
  prepare_output (output_ndat, true);
}
