/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/Convolution.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Apodization.h"
#include "dsp/ResponseProduct.h"
#include "dsp/InputBuffering.h"
#include "dsp/DedispersionHistory.h"
#include "dsp/Dedispersion.h"
#include "dsp/Scratch.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#endif

#include "FTransform.h"

//#define _DEBUG 1
#include "debug.h"

#include <string.h>

using namespace std;

dsp::Convolution::Convolution (const char* _name, Behaviour _type)
  : Transformation<TimeSeries,TimeSeries> (_name, _type)
{
  set_buffering_policy (new InputBuffering (this));
  normalizer = new ScalarFilter();

  zero_DM = false;
  zero_DM_output = new dsp::TimeSeries;
  zero_DM_response  = NULL;

  matrix_convolution = false;
  critically_sampled_ouptut = false;
}

dsp::Convolution::~Convolution ()
{
}

//! Set the device memory to use
void dsp::Convolution::set_device (Memory* mem)
{
  memory = mem;

#if HAVE_CUDA
  CUDA::DeviceMemory* device_memory = dynamic_cast< CUDA::DeviceMemory*> ( mem);

  if ( device_memory )
  {
    Scratch* gpu_scratch = new Scratch;
    gpu_scratch->set_memory (device_memory);
    set_scratch (gpu_scratch);
  }
#endif

}

void dsp::Convolution::set_engine (Engine * _engine)
{
  engine = _engine;

  if (engine)
  {
    engine->set_cerr (this->cerr);
  }
}

void dsp::Convolution::set_cerr (ostream& os) const
{
  if (engine)
    engine->set_cerr(os);
  Transformation<TimeSeries,TimeSeries>::set_cerr(os);
}

dsp::Convolution::Engine* dsp::Convolution::get_engine ()
{
  return engine;
}

//! Set the frequency response function
void dsp::Convolution::set_response (Response* _response)
{
  response = _response;
}

bool dsp::Convolution::has_response () const
{
  return response;
}

const dsp::Response* dsp::Convolution::get_response() const
{
  return response;
}

dsp::Response* dsp::Convolution::get_response()
{
  return response;
}

bool dsp::Convolution::has_passband () const
{
  return passband;
}

const dsp::Response* dsp::Convolution::get_passband() const
{
  return passband;
}

dsp::Response* dsp::Convolution::get_passband()
{
  return passband;
}

const dsp::Apodization* dsp::Convolution::get_temporal_apodization () const
{
  return temporal_apodization;
}

dsp::Apodization* dsp::Convolution::get_temporal_apodization ()
{
  return temporal_apodization;
}


bool dsp::Convolution::has_temporal_apodization () const
{
  return temporal_apodization;
}

void dsp::Convolution::set_temporal_apodization (Apodization* _function)
{
  temporal_apodization = _function;
}

const dsp::Apodization* dsp::Convolution::get_spectral_apodization () const
{
  return spectral_apodization;
}

dsp::Apodization* dsp::Convolution::get_spectral_apodization ()
{
  return spectral_apodization;
}

bool dsp::Convolution::has_spectral_apodization () const
{
  return spectral_apodization;
}

void dsp::Convolution::set_spectral_apodization (Apodization* _function)
{
  spectral_apodization = _function;
}

//! Set the passband integrator
void dsp::Convolution::set_passband (Response* _passband)
{
  passband = _passband;
}

void dsp::Convolution::set_zero_DM_output (TimeSeries* _zero_DM_output)
{
  zero_DM = true;
  zero_DM_output = _zero_DM_output;
}

bool dsp::Convolution::has_zero_DM_output () const
{
  return zero_DM_output;
}

const dsp::TimeSeries* dsp::Convolution::get_zero_DM_output () const
{
  return zero_DM_output;
}

dsp::TimeSeries* dsp::Convolution::get_zero_DM_output ()
{
  return zero_DM_output;
}

bool dsp::Convolution::has_zero_DM_response () const
{
  return zero_DM_response;
}

const dsp::Response* dsp::Convolution::get_zero_DM_response() const
{
  return zero_DM_response;
}

dsp::Response* dsp::Convolution::get_zero_DM_response()
{
  return zero_DM_response;
}

void dsp::Convolution::set_zero_DM_response (dsp::Response* response)
{
  zero_DM_response = response;
}

void dsp::Convolution::set_critically_sampled_output (bool flag)
{
  critically_sampled_ouptut = flag;
}
    
//! Prepare temporal apodization / tapering window
void dsp::Convolution::prepare_temporal_apodization ()
{
  if (!temporal_apodization)
    return;

  if (verbose)
    cerr << "dsp::Convolution::prepare_temporal_apodization" << endl;

  Signal::State state = input->get_state();

  temporal_apodization -> set_size( nsamp_fft );
  temporal_apodization -> set_analytic( state == Signal::Analytic );
  temporal_apodization -> set_transition_start( nfilt_pos );
  temporal_apodization -> set_transition_end( nfilt_neg );
  temporal_apodization -> build ();

  if (temporal_apodization->get_ndat() != nsamp_fft)
    throw Error (InvalidState, "dsp::Convolution::prepare_temporal_apodization",
                 "invalid apodization function ndat=%d"
                 " (nfft=%d)", temporal_apodization->get_ndat(), nsamp_fft);

  if (state == Signal::Analytic && !temporal_apodization->get_analytic())
    throw Error (InvalidState, "dsp::Convolution::prepare_temporal_apodization",
                 "Tapering function not configured for analytic signal.");

  if (state == Signal::Nyquist && temporal_apodization->get_analytic())
    throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                 "Tapering function not configured for real signal.");
} 

//! Prepare spectral apodization / tapering window
void dsp::Convolution::prepare_spectral_apodization ( unsigned bc_nfft )
{
  prepare_spectral_apodization (bc_nfft, input->get_dual_sideband());
}

//! Prepare spectral apodization / tapering window
void dsp::Convolution::prepare_spectral_apodization (unsigned bc_nfft, bool dual_sideband)
{
  if (!spectral_apodization)
    return;

  unsigned critical_nfft = bc_nfft;
  auto os = input->get_oversampling_factor();

  if (critically_sampled_ouptut && os > 1.0)
  {
    critical_nfft = bc_nfft / os;
  }

  if (verbose)
    cerr << "dsp::Convolution::prepare_spectral_apodization"
            " (bcc) nfft=" << bc_nfft << " os=" << os << " nfft_crit=" << critical_nfft << endl;

  spectral_apodization -> set_size( bc_nfft );
  spectral_apodization -> set_analytic( true );
  spectral_apodization -> build ();

  if (dual_sideband)
  {
    if (verbose)
      cerr << "dsp::Convolution::prepare_spectral_apodization swapping for DSB input" << endl;
    spectral_apodization -> swap ();
  }

  spectral_apodization -> dump ("spectral_taper.txt");

  if (spectral_apodization->get_ndat() != bc_nfft)
    throw Error (InvalidState, "Convolution::prepare_spectral_apodization",
                 "invalid apodization function ndat=%d"
                 " (nfft=%d)", spectral_apodization->get_ndat(), bc_nfft);

  if (!spectral_apodization->get_analytic())
    throw Error (InvalidState, "Convolution::prepare_spectral_apodization",
                 "tapering function not configured for complex-valued data");
}

//! Prepare scalar normalizing filters
void dsp::Convolution::prepare_normalizer(double normfac)
{
  normalizer->set_scale_factor (normfac);
  append_response (response, normalizer);

  if (zero_DM)
  {
    if (!zero_DM_normalizer)
      zero_DM_normalizer = new ScalarFilter;

    zero_DM_normalizer->set_scale_factor (normfac);
    append_response (zero_DM_response, zero_DM_normalizer);
  }
}

//! Prepare all relevant attributes
void dsp::Convolution::prepare ()
{
  if (prepared)
    return;

  if (!response)
    throw Error (InvalidState, "dsp::Convolution::prepare",
                 "no frequency response");

  if (input->get_detected())
    throw Error (InvalidState, "dsp::Convolution::prepare",
                 "input data are detected");

  response->prepare (input);

  // zero_DM response should at least have a normalizer
  if (zero_DM && zero_DM_response)
    zero_DM_response->prepare (input);

  // response must have at least two points in it
  if (response->get_ndat() < 2)
    throw Error (InvalidState, "dsp::Convolution::prepare",
                 "invalid response size = %u", response->get_ndat());

  // if the response has 8 dimensions, then perform matrix convolution
  matrix_convolution = response->get_ndim() == 8;

  Signal::State state = input->get_state();
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();

  // if matrix convolution, then there must be two polns
  if (matrix_convolution && npol != 2)
    throw Error (InvalidState, "dsp::Convolution::prepare",
                 "matrix response and input.npol != 2");

  // response must contain a unique kernel for each channel
  if (response->get_nchan() != nchan)
    throw Error (InvalidState, "dsp::Convolution::prepare",
                 "invalid response nsub=%d != nchan=%d",
                 response->get_nchan(), nchan);

  // number of points after first fft
  n_fft = response->get_ndat();

  //! Complex samples dropped from beginning of cyclical convolution result
  nfilt_pos = response->get_impulse_pos ();

  //! Complex samples dropped from end of cyclical convolution result
  nfilt_neg = response->get_impulse_neg ();

  nfilt_tot = nfilt_pos + nfilt_neg;

  if (verbose)
    cerr << "Convolution::prepare filt=" << n_fft
         << " smear=" << nfilt_tot << endl;

  // 2 arrays needed: one for each of the forward and backward FFT results
  // 2 floats per complex number
  scratch_needed = n_fft * 2 * 2;

  if (matrix_convolution)
    // need space for one more complex spectrum
    scratch_needed += n_fft * 2;

  // number of time samples in forward fft and overlap region
  nsamp_fft = 0;
  nsamp_overlap = 0;

  if (state == Signal::Nyquist)
  {
    nsamp_fft = n_fft * 2;
    nsamp_overlap = nfilt_tot * 2;
    scratch_needed += 4;
  }
  else if (state == Signal::Analytic)
  {
    nsamp_fft = n_fft;
    nsamp_overlap = nfilt_tot;
  }
  else
    throw Error (InvalidState, "dsp::Convolution::prepare",
                 "Cannot transform Signal::State="
                 + tostring(input->get_state()));

  scalefac = 1.0;
  if (FTransform::get_norm() == FTransform::unnormalized)
  {
    scalefac = double(nsamp_fft);

    // the backward FFT is half the length
    if (state == Signal::Nyquist)
      scalefac /= 2.0;
  }

  // sqrt because the scale factor is applied prior to detection
  scalefac = sqrt(scalefac);
  prepare_normalizer (1.0/scalefac);
  scalefac = 1.0;

  if (temporal_apodization)
    prepare_temporal_apodization ();

  if (response)
    response->prepare(input);

  if (zero_DM_response)
    zero_DM_response->prepare (input);

  // the FFT size must be greater than the number of discarded points
  if (nsamp_fft < nsamp_overlap)
    throw Error (InvalidState, "dsp::Convolution::prepare",
                 "error nfft=%d < nfilt=%d", nsamp_fft, nsamp_overlap);

  if (has_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::Convolution::prepare reserve=" << nsamp_fft << endl;

    get_buffering_policy()->set_maximum_samples (nsamp_fft);
  }
  if (zero_DM) {
    if (!zero_DM) {
      zero_DM_output = new dsp::TimeSeries;
    }
  }

  prepare_output ();
  prepare_passband ();

  if (engine)
  {
    if (verbose)
      cerr << "dsp::Convolution::make_preparations setup engine" << endl;
    engine->prepare (this);
    prepared = true;
    return;
  }

  using namespace FTransform;

  if (state == Signal::Nyquist)
    forward = Agent::current->get_plan (nsamp_fft, FTransform::frc);
  else
    forward = Agent::current->get_plan (nsamp_fft, FTransform::fcc);

  backward = Agent::current->get_plan (n_fft, FTransform::bcc);

  prepared = true;
}

void dsp::Convolution::prepare_output ()
{
  Signal::State state = input->get_state();
  const uint64_t ndat = input->get_ndat();

  if (verbose)
    cerr << "dsp::Convolution::prepare_output input"
         << " ndat=" << input->get_ndat()
         << " state=" << input->get_state() << endl;

  // valid time samples per FFT
  nsamp_step = nsamp_fft-nsamp_overlap;

  if (verbose)
    cerr << "dsp::Convolution::prepare_output nsamp fft=" << nsamp_fft
         << " overlap=" << nsamp_overlap << " step=" << nsamp_step << endl;

  // number of FFTs for this data block
  npart = 0;
  if (ndat >= nsamp_fft)
    npart = (ndat-nsamp_overlap)/nsamp_step;

  if (engine)
  {
    //scratch_needed = npart * n_fft * 2;
    scratch_needed = n_fft * 2 * 2;
  }

  /*
    The input must be buffered before the output is modified
    because the transformation may be inplace
  */
  if (has_buffering_policy() && input->get_input_sample() >= 0)
    get_buffering_policy()->set_next_start (nsamp_step * npart);
#if DEBUGGING_OVERLAP
  // this exception is useful when debugging, but not at the end-of-file
  else if (ndat > 0 && (nsamp_step * npart + nsamp_overlap != ndat))
    throw Error (InvalidState, "dsp::Convolution::prepare_output",
                 "npart=%u * step=%u + overlap=%u != ndat=%u",
		 npart, nsamp_step, nsamp_overlap, ndat);
#endif

  // prepare the output TimeSeries
  output->copy_configuration (input);

  output->set_state( Signal::Analytic );
  output->set_ndim( 2 );

  if ( state == Signal::Nyquist ) {
    output->set_rate( 0.5*get_input()->get_rate() );
  }
  // set the input sample
  uint64_t output_ndat = output->get_ndat();
  int64_t input_sample = input->get_input_sample();
  if (output_ndat == 0)
    output->set_input_sample (0);
  else if (input_sample >= 0)
    output->set_input_sample ((input_sample / nsamp_step) * nsamp_step);

  if (zero_DM) {
    zero_DM_output->copy_configuration(output);
    zero_DM_output->set_input_sample(output->get_input_sample());
    if (verbose) {
      std::cerr << "dsp::Convolution::prepare: output->get_nchan()=" <<
        output->get_nchan() << ",  zero_DM_output->get_nchan()=" <<
        zero_DM_output->get_nchan() << std::endl;
      std::cerr << "dsp::Convolution::prepare: output->get_npol()=" <<
        output->get_npol() << ",  zero_DM_output->get_npol()=" <<
        zero_DM_output->get_npol() << std::endl;
      std::cerr << "dsp::Convolution::prepare: output->get_ndat()=" <<
        output->get_ndat() << ",  zero_DM_output->get_ndat()=" <<
        zero_DM_output->get_ndat() << std::endl;
      std::cerr << "dsp::Convolution::prepare: output->get_state()=" <<
        output->get_state() << ",  zero_DM_output->get_state()=" <<
        zero_DM_output->get_state() << std::endl;
      std::cerr << "dsp::Convolution::prepare: output->get_ndim()=" <<
        output->get_ndim() << ",  zero_DM_output->get_ndim()=" <<
        zero_DM_output->get_ndim() << std::endl;
      std::cerr << "dsp::Convolution::prepare: output->get_rate()=" <<
        output->get_rate() << ",  zero_DM_output->get_rate()=" <<
        zero_DM_output->get_rate() << std::endl;
      std::cerr << "dsp::Convolution::prepare: output->get_input_sample()=" <<
        output->get_input_sample() << ",  zero_DM_output->get_input_sample()=" <<
        zero_DM_output->get_input_sample() << std::endl;
    }
  }
}

void dsp::Convolution::prepare_passband ()
{
  // the engine should delete the passband if it doesn't support this feature
  if (!passband)
    return;

  if (response)
    passband->match_shape (response);

  unsigned passband_npol = input->get_npol();
  if (matrix_convolution)
    passband_npol = 4;

  passband->resize (passband_npol, input->get_nchan(), n_fft, 1);

  if (!response)
    passband->match (input);

  passband->zero();
}

//! Reserve the maximum amount of output space required
void dsp::Convolution::reserve ()
{
  const uint64_t ndat = input->get_ndat();
  Signal::State state = input->get_state();

  prepare_output ();

  if (verbose)
    cerr << "Convolution::reserve ndat=" << ndat << " nfft=" << nsamp_fft
         << " npart=" << npart << endl;

  WeightedTimeSeries* weighted_output;
  weighted_output = dynamic_cast<WeightedTimeSeries*> (output.get());
  if (weighted_output)
  {
    if (verbose)
      cerr << "dsp::Convolution::reserve calling "
              "WeightedTimeSeries::convolve_weights" << endl;

    weighted_output->convolve_weights (nsamp_fft, nsamp_step);
    if (state == Signal::Nyquist)
      weighted_output->scrunch_weights (2);
  }

  uint64_t output_ndat = npart * nsamp_step;
  if ( state == Signal::Nyquist )
    output_ndat /= 2;

  if (input != output)
    output->resize (output_ndat);
  else
    output->set_ndat (output_ndat);
  
  // nfilt_pos complex points are dropped from the start of the first FFT
  output->change_start_time (nfilt_pos);

  // data will be normalised by the response product, removing the effect
  // of an unnormalized FFT
  if (verbose)
    cerr << "Convolution::reserve scale="<< output->get_scale() <<endl;

  if (zero_DM)
  {
    zero_DM_output->resize(output_ndat);
    zero_DM_output->change_start_time(nfilt_pos);
  }

  if (verbose)
    cerr << "Convolution::reserve done" << endl;
}

double dsp::Convolution::get_delay_time () const
{
  return nfilt_pos / output->get_rate();
}

/*!
  \pre input TimeSeries must contain phase coherent (undetected) data
  \post output TimeSeries will contain complex (Analytic) data

  \post IMPORTANT!! Most backward complex FFT functions expect
  frequency components organized with f0+bw/2 -> f0, f0-bw/2 -> f0.
  The forward real-to-complex FFT produces f0-bw/2 -> f0+bw/2.  To
  save CPU cycles, convolve() does not re-sort the ouput array, and
  therefore introduces a frequency shift in the output data.  This
  results in a phase gradient in the time domain.  Since only
  relative phases matter when calculating the Stokes parameters,
  this effect is basically ignorable for our purposes.
*/
void dsp::Convolution::transformation ()
{
  Signal::State state  = input->get_state();
  const unsigned npol  = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim  = input->get_ndim();

  if (!prepared)
    prepare ();

  if (response)
  {
    if (verbose)
      cerr << "dsp::Convolution::transformation call Response::match" << endl;
    bool changed = response -> match (input, nchan);
    if (changed && engine)
    {
      if (verbose)
        cerr << "dsp::Convolution::transformation call Engine::copy_response" << endl;
      engine->copy_response(response);
    }
  }

  reserve ();

  if (verbose)
    cerr << "dsp::Convolution::transformation after calling reserve"
         << "\n\t input ndat=" << input->get_ndat()
         << "\n\t output ndat=" << output->get_ndat() << endl;

  if (verbose)
  {
    cerr << "dsp::Convolution::transformation scratch"
      " size=" << scratch_needed  << endl;
    if (zero_DM)
      cerr << "dsp::Convolution::transformation using zero DM output" << endl;
  }

  float* spectrum[2];
  spectrum[0] = scratch->space<float> (scratch_needed);
  spectrum[1] = spectrum[0];
  if (matrix_convolution)
    spectrum[1] += n_fft * 2;

  if (engine)
  {
    engine->set_scratch (spectrum[0]);
    if (zero_DM)
      engine->perform (input, output, zero_DM_output, npart);
    else
      engine->perform (input, output, npart);
    return;
  }

  float* complex_time = spectrum[1] + n_fft * 2;

  // although only two extra points are required, adding 4 ensures that SIMD alignment is maintained
  if (state == Signal::Nyquist)
    complex_time += 4;

  const unsigned nbytes_step = nsamp_step * ndim * sizeof(float);

  if (verbose) 
  {
    cerr << "dsp::Convolution::transformation step nsamp=" << nsamp_step
         << " bytes=" << nbytes_step << " ndim=" << ndim << " nfilt_pos=" << nfilt_pos << endl;
  }

  const unsigned cross_pol = matrix_convolution ? 2 : 1;

  // temporary things that should not go in and out of scope
  float* ptr = 0;
  unsigned jpol=0;

  uint64_t offset;
  // number of floats to step between each FFT
  const uint64_t step = nsamp_step * ndim;

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      for (uint64_t ipart=0; ipart < npart; ipart++)
      {
        offset = ipart * step;

        for (jpol=0; jpol<cross_pol; jpol++)
        {
          if (matrix_convolution)
            ipol = jpol;

          ptr = const_cast<float*>(input->get_datptr (ichan, ipol)) + offset;

          if (temporal_apodization)
          {
            temporal_apodization -> operate (ptr, complex_time);
            ptr = complex_time;
          }

          DEBUG("FORWARD: nfft=" << nsamp_fft << " in=" << ptr \
                << " out=" << spectrum[ipol]);

          if (state == Signal::Nyquist)
            forward->frc1d (nsamp_fft, spectrum[ipol], ptr);

          else if (state == Signal::Analytic)
            forward->fcc1d (nsamp_fft, spectrum[ipol], ptr);

        } // for each poln, if matrix convolution

        if (matrix_convolution)
        {
          response->operate (spectrum[0], spectrum[1], ichan);

          if (passband)
            passband->integrate (spectrum[0], spectrum[1], ichan);
        }

        else
        {
          response->operate (spectrum[ipol], ipol, ichan);

          if (passband)
            passband->integrate (spectrum[ipol], ipol, ichan);
        }

        for (jpol=0; jpol<cross_pol; jpol++)
        {
          if (matrix_convolution)
            ipol = jpol;

          DEBUG("BACKWARD: nfft=" << n_fft << " in=" << spectrum[ipol] \
                << " out=" << complex_time);

          // fft back to the complex time domain
          backward->bcc1d (n_fft, complex_time, spectrum[ipol]);

          // copy the good (complex) data back into the time stream
          ptr = output -> get_datptr (ichan, ipol) + offset;

          DEBUG("memcpy: nbytes=" << nbytes_step \
                << " in=" << complex_time + nfilt_pos*2 \
                << " out=" << ptr << " offset=" << offset);

          memcpy (ptr, complex_time + nfilt_pos*2, nbytes_step);

          if (zero_DM)
          {
            memcpy(
              zero_DM_output->get_datptr(ichan, ipol) + offset,
              input->get_datptr(ichan, ipol) + offset + nfilt_pos*2,
              nbytes_step
            );
          }
        }  // for each poln, if matrix convolution
      }  // for each part of the time series
    }  // for each poln
  }  // for each channel

  if (verbose)
  {
    std::cerr << "dsp::Convolution::transformation: output->get_input_sample()=" <<
      output->get_input_sample() << ",  zero_DM_output->get_input_sample()=" <<
      zero_DM_output->get_input_sample() << std::endl;
  }


}
