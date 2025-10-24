/***************************************************************************
 *
 *   Copyright (C) 2002 - 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Response.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "Error.h"
#include "Jones.h"
#include "cross_detect.h"

#include <vector>
#include <assert.h>
#include <math.h>

using namespace std;

// #define _DEBUG

unsigned dsp::Response::ndat_max = 0;

dsp::Response::Response ()
{
  // assume complex-valued, single polarization, single frequency channel
  ndim = 2;
  npol = 1;
  nchan = 1;
}

dsp::Response::~Response () {}

const dsp::Response& dsp::Response::operator = (const Response& response)
{
  if (this == &response)
    return *this;

  copy (&response);

  return *this;
}

void dsp::Response::copy (const Response* response)
{
  Shape::operator = ( *response );

  input_nchan = response->input_nchan;
  impulse_pos = response->impulse_pos;
  impulse_neg = response->impulse_neg;
  whole_swapped = response->whole_swapped;
  swap_divisions = response->swap_divisions;
  dc_centred = response->dc_centred;
  step = response->step;
}

void dsp::Response::changed ()
{
  if (verbose)
    cerr << "dsp::Response::changed" << endl;

  built = false;
}

void dsp::Response::build (const Observation* input) 
{
  if (verbose)
    cerr << "dsp::Response::build setting all values to zero" << endl;
  zero();
}

void dsp::Response::set_frequency_resolution (unsigned nfft)
{
  if (verbose)
    cerr << "dsp::Response::set_frequency_resolution ("<<nfft<<")"<<endl;

  user_set_frequency_resolution = nfft;
  changed();
}

void dsp::Response::set_times_minimum_nfft (unsigned times)
{
  if (verbose)
    cerr << "dsp::Response::set_times_minimum_nfft ("<<times<<")"<<endl;

  user_set_times_minimum_nfft = times;
  changed();
}

void dsp::Response::set_impulse_samples (unsigned tot)
{
  unsigned pos = tot / 2;
  unsigned neg = pos;
  if (tot % 2)
    neg ++;
  set_impulse_samples (pos, neg);
}

void dsp::Response::set_impulse_samples (unsigned pos, unsigned neg)
{
  user_set_impulse_pos = pos;
  user_set_impulse_neg = neg;
  changed();
}

void dsp::Response::set_downsampling (bool flag)
{
  if (flag != downsampling)
    changed();
  downsampling = flag;
}

void dsp::Response::multiply (const Response* response)
{
  if (verbose)
    cerr << "dsp::Response::multiply" << endl;

  if (ndat*nchan != response->ndat * response->nchan)
    throw Error (InvalidParam, "dsp::Response::multiply",
          "this ndat=%u*nchan=%u != other ndat=%u*nchan=%u", ndat, nchan,
          response->ndat, response->nchan);

  if (npol < response->npol)
    throw Error (InvalidParam, "dsp::Response::multiply",
          "this.npol=%d < other.npol=%d", npol, response->npol);

  /*
    perform A = A * B where
    A = this->buffer
    B = response->buffer
    (so B operates on A's buffer)
  */

  if (response->ndim > ndim)
    throw Error (InvalidParam, "dsp::Response::multiply",
          "this.ndim=%u >= other.ndim=%u , incorrect ndim order for operation",
          ndim, response->ndim);

  unsigned original_step = response->step;
  response->step = ndim / response->ndim;

#ifdef _DEBUG
  cerr << "dsp::Response::multiply npol=" << npol << " nchan=" << nchan
       << " this.ndim=" << ndim << " other.ndim=" << response->ndim
       << " this.step=" << step << " other.step=" << response->step << endl;
#endif

  for (unsigned istep=0; istep < response->step; istep++)
    for (unsigned ipol=0; ipol < npol; ipol++)
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        float* ptr = get_datptr (ichan, ipol);
        response->operate (ptr + istep*response->ndim, ipol, ichan);
      }

  response->step = original_step;

  changed();

#ifdef _DEBUG
  cerr << "dsp::Response::multiply return" << endl;
#endif
}

void dsp::Response::configure (const Observation* input, unsigned channels)
{
  impulse_neg = impulse_pos = 0;
}

bool dsp::Response::match (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Response::match channels=" << channels << endl;

  prepare (input, channels);
  
  if (!has_changed())
  {
    if (verbose)
      cerr << "dsp::Response::match response has not changed - returning false" << endl;    
    return false;
  }

  if (verbose)
    cerr << "dsp::Response::match response has changed - rebuilding" << endl;
  rebuild(input);
  return true;
}

void dsp::Response::rebuild (const Observation* input)
{
  currently_building = true;

  whole_swapped = false;
  swap_divisions = 0;
  dc_centred = false;

  resize_and_build (input);
  if (input)
    swap_as_needed (input);

  built = true;
  currently_building = false;
}

void dsp::Response::prepare (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Response::prepare channels=" << channels << endl;

  configure (input, channels);

  if (channels > 0)
    set_nchan(channels);
  else if (input)
    set_nchan(input->get_nchan());

  if (!has_changed())
  {
    if (verbose)
      cerr << "dsp::Response::prepare nothing changed - returning" << endl;
    return;
  }

  if (verbose)
    cerr << "dsp::Response::prepare response has changed - setting ndat" << endl;

  if (input && downsampling)
  {
    prepare_downsampling(input);
  }

  if (user_set_times_minimum_nfft)
  {
    if (verbose)
      cerr << "dsp::Response::prepare user-specified multiple of minimum frequency resolution = " << user_set_times_minimum_nfft << endl;

    set_ndat( user_set_times_minimum_nfft * get_minimum_ndat() );
  }
  else if (user_set_frequency_resolution)
  {
    if (verbose)
      cerr << "dsp::Response::prepare user-specified frequency resolution = " << user_set_frequency_resolution << endl;

    set_ndat(user_set_frequency_resolution);
  }
  else
  {
    if (optimal_fft)
      optimal_fft->set_simultaneous (nchan > 1);

    unsigned multiple = 1;

    if (input && downsampling)
      multiple = downsampling_factor;

    set_optimal_ndat (multiple);
  }

  if (input && downsampling)
  {
    // ensure that ndat is an integer in oversampled input
    if (ndat % downsampling_factor)
      throw Error (InvalidState, "dsp::Response::prepare",
                  "ndat=%u is not a multiple of downsampling_factor=%u", ndat, downsampling_factor);
  }
  if (verbose)
  {
    std::cerr << "dsp::Response::prepare this=" << (void*) this
      << " ndat=" << ndat
      << " impulse_pos=" << impulse_pos
      << " impulse_neg=" << impulse_neg
      << std::endl;
  }
}


// returns the smallest multiple of A greater than or equal to B
// T should be of integer type
template<typename T>
T smallest_multiple(T A, T B)
{
  // a * ceil(b/a)
  return A * ((A + B - 1) / A);
}

void dsp::Response::prepare_downsampling (const Observation* input)
{
  auto osf = input->get_oversampling_factor();

  if (verbose)
    cerr << "dsp::Response::prepare_downsampling over_sampling_factor=" << osf << endl;    

  downsampling_factor = osf.get_denominator();
  
  if (input->get_nchan() > get_nchan())
  {
    if (verbose)
      cerr << "dsp::Response::prepare_downsampling decimating from " << input->get_nchan() << " to " << get_nchan() << endl;

    if (input->get_nchan() % get_nchan())
    {
      throw Error (InvalidState, "dsp::Response::prepare_downsampling",
                  "input nchan=%d is not divisible by nchan=%d",
                  input->get_nchan(), get_nchan());
    }

    downsampling_factor *= input->get_nchan() / get_nchan();
  }

  if (verbose)
    cerr << "dsp::Response::prepare_downsampling setting impulse_pos=" << impulse_pos << " and impulse_neg=" << impulse_neg 
          << " to smallest possible multiples of " << downsampling_factor << endl;

  impulse_pos = smallest_multiple(downsampling_factor, impulse_pos);
  impulse_neg = smallest_multiple(downsampling_factor, impulse_neg);

  if (verbose)
    cerr << "dsp::Response::prepare_downsampling resulting impulse_pos=" << impulse_pos 
         << " and impulse_neg=" << impulse_neg << endl;
}

void dsp::Response::swap_as_needed (const Observation* input) try
{
  if (!input)
    throw Error(InvalidState, "dsp::Response::swap_as_needed", "input == nullptr");

  if (verbose)
    cerr << "dsp::Response::swap_as_needed input.nchan=" << input->get_nchan() << endl;

  input_nchan = input->get_nchan();

  if ( input_nchan == 1 )
  {
    // if the input Observation is single-channel, complex sampled
    // data, then the first forward FFT performed on this data will
    // result in a swapped spectrum
    if ( input->get_dual_sideband() > 0 && !whole_swapped )
    {
      if (verbose)
        cerr << "dsp::Response::swap_as_needed swap whole" << endl;
      doswap ();
    }

    unsigned subswap = abs(input->get_dual_sideband());

    if ( subswap > 1 && swap_divisions != subswap )
    {
      if (verbose)
        cerr << "dsp::Response::swap_as_needed subswap=" << subswap << endl;
      doswap(subswap);
    }
  }
  else
  {
    // if the filterbank channels are centred on DC
    if ( input->get_dc_centred() && !dc_centred )
    {
      if (verbose)
        cerr << "dsp::Response::swap_as_needed rotate half channel" << endl;

      if ( swap_divisions )
        doswap ( swap_divisions );

      rotate (-int(ndat/2));
      dc_centred = true;
    }

    /* If the input Observation is multi-channel, complex-valued data with dual-sideband 
       spectral order in each channel, and this response has the same number of channels,
       and the data were not already swapped at this frequency channel resolution,
       then each FFT performed will result in little swapped spectra.

       This response will not have the same number of channels as the input when either
       a) the input will be further channelized (e.g. Filterbank)
       b) input channels will be synthesized into wider ones (e.g. InverseFilterbank)
    */
    if (input->get_dual_sideband() && input->get_nchan() == get_nchan() && get_nchan() != swap_divisions)
    {
      if (verbose)
        cerr << "dsp::Response::swap_as_needed swap channels"
                " (nchan=" << get_nchan() << ")" << endl;
      doswap ( get_nchan() );
    }

    // the ordering of the filterbank channels may be swapped
    if ( input->get_swap() && !whole_swapped )
    {
      if (verbose)
        cerr << "dsp::Response::swap_as_needed swap whole (nchan=" << nchan << ")" << endl;
      doswap ();
    }
  }
}
catch (Error& error)
{
  throw error += "dsp::Response::swap_as_needed";
}

void dsp::Response::resize_and_build (const Observation* input) try
{
  ThreadContext::Lock lock (context);

  if (built)
  {
    if (verbose)
      cerr << "dsp::Response::resize_and_build already built - returning" << endl;
    return;
  }

  check_ndat ();
  resize (npol, nchan, ndat, ndim);
  verify_dataspace();
  build (input);

  // set DC to zero only for real-valued input
  if (input && input->get_state() == Signal::Nyquist)
    buffer[0] = buffer[1] = 0.0;
}
catch (Error& error)
{
  throw error += "dsp::Response::resize_and_build";
}

//! Returns true if the dimension and ordering match
bool dsp::Response::matches (const Shape* shape)
{
  const Response* response = dynamic_cast<const Response*> (shape);

  if (!response)
    return false;

  return
    whole_swapped == response->whole_swapped &&
    swap_divisions == response->swap_divisions &&
    dc_centred == response->dc_centred &&

    nchan == response->get_nchan() &&
    ndat == response->get_ndat();
}

void dsp::Response::match_shape (const Response* response)
{
  if (verbose)
    cerr << "dsp::Response::match_shape Response" << endl;

  resize (npol, response->get_nchan(), response->get_ndat(), ndim);

  input_nchan = response->input_nchan;
  whole_swapped = response->whole_swapped;
  swap_divisions = response->swap_divisions;
  dc_centred = response->dc_centred;
}


//! Set the dimensions of the data and update the built attribute
void dsp::Response::resize (unsigned _npol, unsigned _nchan, unsigned _ndat, unsigned _ndim)
{
  if (verbose)
    cerr << "dsp::Response::resize npol=" << _npol << " nchan=" << _nchan
         << " ndat=" << _ndat << " ndim=" << _ndim << endl;

  if (npol != _npol || nchan != _nchan || ndat != _ndat || ndim != _ndim)
  {
    changed();
  }
  Shape::resize (_npol, _nchan, _ndat, _ndim);
}

//! Set the length of the frequency response in each channel
void dsp::Response::set_ndat (unsigned _ndat)
{
  if (ndat != _ndat)
    changed();
  ndat = _ndat;
}

//! Set the number of input channels 
void dsp::Response::set_nchan (unsigned _nchan)
{
  if (nchan != _nchan)
    changed();
  nchan = _nchan;
}

//! Set the flag for a bin-centred spectrum
void dsp::Response::set_dc_centred (bool _dc_centred)
{
  if (dc_centred != _dc_centred)
    changed();

  dc_centred = _dc_centred;
}

void dsp::Response::naturalize ()
{
  if (verbose)
    cerr << "dsp::Response::naturalize" << endl;

  if ( whole_swapped )
  {
    if (verbose)
      cerr << "dsp::Response::naturalize whole bandpass swap" << endl;
    doswap ();
  }

  if ( swap_divisions )
  {
    if (verbose)
      cerr << "dsp::Response::naturalize sub-bandpass swap" << endl;
    doswap ( swap_divisions );
  }

  if ( dc_centred )
  {
    if (verbose)
      cerr << "dsp::Response::naturalize rotation" << endl;
    rotate (ndat/2);
    dc_centred = false;
  }
}

/*!  Using the impulse_pos and impulse_neg attributes, this method
  determines the minimum acceptable ndat for use in convolution.  This
  is given by the smallest power of two greater than or equal to the
  twice the sum of impulse_pos and impulse_neg. */
unsigned dsp::Response::get_minimum_ndat () const
{
  double impulse_tot = impulse_pos + impulse_neg;

  if (impulse_tot == 0)
    return 0;

  unsigned min = unsigned( pow (2.0, ceil( log(impulse_tot)/log(2.0) )) );
  while (min <= impulse_tot)
    min *= 2;

  if (verbose)
    cerr << "dsp::Response::get_minimum_ndat impulse_tot=" << impulse_tot
         << " min power of two=" << min << endl;

  return min;
}

// defined in optimize_fft.c
extern "C" int64_t
optimal_fft_length (uint64_t nbadperfft, uint64_t nfft_max, unsigned factor, char verbose);

/*!  Using the get_minimum_ndat method and the max_ndat static attribute,
  this method determines the optimal ndat for use in convolution. */
void dsp::Response::set_optimal_ndat (unsigned multiple)
{
  unsigned ndat_min = impulse_pos + impulse_neg;

  if (verbose)
    cerr << "dsp::Response::set_optimal_ndat ndat_min=" << ndat_min << " multiple=" << multiple << endl;

  if (ndat_min == 0)
    return;

  if (ndat_max && ndat_max < ndat_min)
    throw Error (InvalidState, "dsp::Response::set_optimal_ndat",
          "specified maximum ndat (%d) < required minimum ndat (%d)",
          ndat_max, ndat_min);

  int64_t optimal_ndat = 0;

  if (optimal_fft)
  {
    optimal_fft->set_nchan (nchan);
    optimal_ndat = optimal_fft->get_nfft (ndat_min);
    if (verbose)
      cerr << "dsp::Response::set_optimal_ndat optimal_fft strategy returns optimal_ndat=" << optimal_ndat << endl;
  }
  else
  {
    optimal_ndat = optimal_fft_length (ndat_min, ndat_max, multiple, verbose);
    if (verbose)
      cerr << "dsp::Response::set_optimal_ndat optimal_fft_length function returns optimal_ndat=" << optimal_ndat << endl;
  }

  if (optimal_ndat < 0)
    throw Error (InvalidState, "dsp::Response::set_optimal_ndat", "optimal_ndat < 0");

  ndat = optimal_ndat;
}

void dsp::Response::set_optimal_fft (OptimalFFT* policy)
{
  optimal_fft = policy;
}

dsp::OptimalFFT* dsp::Response::get_optimal_fft () const
{
  return optimal_fft;
}

bool dsp::Response::has_optimal_fft () const
{
  return optimal_fft;
}

void dsp::Response::check_ndat () const
{
  if (ndat_max && ndat > ndat_max)
    throw Error (InvalidState, "dsp::Response::check_ndat",
          "specified maximum ndat (%d) < specified ndat (%d)",
          ndat_max, ndat);

  unsigned ndat_min = get_minimum_ndat ();

  if (verbose)
    cerr << "dsp::Response::check_ndat minimum ndat=" << ndat_min << endl;

  if (ndat < ndat_min)
    throw Error (InvalidState, "dsp::Response::check_ndat",
          "specified ndat (%d) < required minimum ndat (%d)",
          ndat, ndat_min);
}

//! Get the passband
vector<float> dsp::Response::get_passband (unsigned ipol, int ichan) const
{
  assert (ndim == 1);

  // output all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  register const float* f_p = get_datptr (ichan, ipol);

  vector<float> retval (npts);
  for (unsigned ipt=0; ipt < npts; ipt++)
    retval[ipt]=f_p[ipt];

  return retval;
}

// /////////////////////////////////////////////////////////////////////////

/*! Multiplies an array of complex points by the complex response

  \param ipol the polarization of the data (Response may optionally
  contain a different frequency response function for each polarization)

  \param data an array of nchan*ndat complex numbers */

void dsp::Response::operate (float* data, unsigned ipol, int ichan) const
{
  if( ichan < 0 )
    operate(data,ipol,ichan,nchan);
  else
    operate(data,ipol,ichan,1);
}

//! Multiply spectrum by complex frequency response in specified channels
void
dsp::Response::operate (float* spectrum, unsigned poln, int ichan_start, 
                        unsigned nchan_op) const
{
  assert (ndim == 2);

  unsigned ipol = poln;

  // one filter may apply to two polns
  if (ipol >= npol)
    ipol = 0;

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan_start < 0) {
    npts *= nchan;
    ichan_start = 0;
  }
  else
    npts *= nchan_op;

  register float* d_from = spectrum;
  register const float* f_p = get_datptr (ichan_start, ipol);

  /*
    this operates on spectrum; i.e.  A = A * B where
    A = spectrum
    B = this->buffer
  */

#ifdef _DEBUG
  cerr << "dsp::Response::operate nchan=" << nchan << " ipol=" << ipol
       << " buf=" << buffer << " f_p=" << f_p
       << " off=" << offset << endl;
#endif

  // the idea is that by explicitly calling the values from the
  // arrays into local stack space, the routine should run faster
  register float d_r;
  register float d_i;
  register float f_r;
  register float f_i;

  // cerr << "dsp::Response::operate step=" << step << endl;

  for (unsigned ipt=0; ipt<npts; ipt++)
  {
    d_r = d_from[0];
    d_i = d_from[1];
    f_r = f_p[0];
    f_i = f_p[1];

#ifdef _DEBUG
    cerr << "d=" << d_r << " +i" << d_i << " f=" << f_r << " +i" << f_i << endl;
#endif

    d_from[0] = f_r * d_r - f_i * d_i;
    d_from[1] = f_i * d_r + f_r * d_i;

    d_from += 2 * step;
    f_p += 2;
  }

  // cerr << "dsp::Response::operate done" << endl;
}

//! Multiply spectrum by complex frequency response
void
dsp::Response::operate (float* input_spectrum, float * output_spectrum,
                        unsigned poln, int ichan_start, unsigned nchan_op) const
{
  assert (ndim == 2);

  unsigned ipol = poln;

  // one filter may apply to two polns
  if (ipol >= npol)
    ipol = 0;

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan_start < 0) {
    npts *= nchan;
    ichan_start = 0;
  }
  else
    npts *= nchan_op;

  register float* d_from = input_spectrum;
  register float* d_into = output_spectrum;
  register const float* f_p = get_datptr (ichan_start, ipol);

  /*
    this operates on spectrum; i.e.  C = A * B where
    A = input_spectrum
    B = this->buffer
    C = output_spectrum
  */

#ifdef _DEBUG
  cerr << "dsp::Response::operate nchan=" << nchan << " ipol=" << ipol
       << " buf=" << buffer << " f_p=" << f_p
       << " off=" << offset << endl;
#endif

  // the idea is that by explicitly calling the values from the
  // arrays into local stack space, the routine should run faster
  register float d_r;
  register float d_i;
  register float f_r;
  register float f_i;

  for (unsigned ipt=0; ipt<npts; ipt++)
  {
    d_r = d_from[0];
    d_i = d_from[1];
    f_r = f_p[0];
    f_i = f_p[1];

    d_into[0] = f_r * d_r - f_i * d_i;
    d_into[1] = f_i * d_r + f_r * d_i;

    d_from += 2 * step;
    d_into += 2 * step;
    f_p += 2;
  }
}

// /////////////////////////////////////////////////////////////////////////

/*! Adds the square of each complex point to the current power spectrum

  \param data an array of nchan*ndat complex numbers

  \param ipol the polarization of the data (Response may optionally
  integrate a different power spectrum for each polarization)

*/
void dsp::Response::integrate (float* data, unsigned ipol, int ichan)
{
  assert (ndim == 1);
  assert (npol != 4);

  // may be used to integrate total intensity from two polns
  if (ipol >= npol)
    ipol = 0;

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0)
  {
    npts *= nchan;
    ichan = 0;
  }

  register float* d_p = data;
  register float* f_p = get_datptr (ichan, ipol);

#ifdef _DEBUG
  cerr << "dsp::Response::integrate ipol=" << ipol
       << " buf=" << buffer << " f_p=" << f_p
       << "off=" << offset << endl;
#endif

  register float d;
  register float t;

  for (unsigned ipt=0; ipt<npts; ipt++)
  {
    d = *d_p; d_p ++; // Re
    t = d*d;
    d = *d_p; d_p ++; // Im

    *f_p += t + d*d;
    f_p ++;
  }
}

void dsp::Response::set (const vector<complex<float> >& filt)
{
  // one poln, one channel, complex
  resize (1, 1, filt.size(), 2);
  float* f = buffer;

  for (unsigned idat=0; idat<filt.size(); idat++) {
    // Re
    *f = filt[idat].real();
    f++;
    // Im
    *f = filt[idat].imag();
    f++;
  }
}

// /////////////////////////////////////////////////////////////////////////
//
// Response::operate - multiplies two complex arrays by complex matrix Response
// ndat = number of complex points
//
void dsp::Response::operate (float* data1, float* data2, int ichan) const
{
  assert (ndim == 8);

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  float* d1_rp = data1;
  float* d1_ip = data1 + 1;
  float* d2_rp = data2;
  float* d2_ip = data2 + 1;

  float* f_p = buffer + ichan * ndat * ndim;

  register float d_r;
  register float d_i;
  register float f_r;
  register float f_i;

  register float r1_r;
  register float r1_i;
  register float r2_r;
  register float r2_i;

  for (unsigned ipt=0; ipt<npts; ipt++)
  {
    // ///////////////////////
    // multiply: r1 = f11 * d1
    d_r = *d1_rp;
    d_i = *d1_ip;
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    r1_r = f_r * d_r - f_i * d_i;
    r1_i = f_i * d_r + f_r * d_i;

    // ///////////////////////
    // multiply: r2 = f21 * d1
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    r2_r = f_r * d_r - f_i * d_i;
    r2_i = f_i * d_r + f_r * d_i;

    // ////////////////////////////
    // multiply: d2 = r2 + f22 * d2
    d_r = *d2_rp;
    d_i = *d2_ip;
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    *d2_rp = r2_r + (f_r * d_r - f_i * d_i);
    d2_rp += 2;
    *d2_ip = r2_i + (f_i * d_r + f_r * d_i);
    d2_ip += 2;

    // ////////////////////////////
    // multiply: d1 = r1 + f12 * d2
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    *d1_rp = r1_r + (f_r * d_r - f_i * d_i);
    d1_rp += 2;
    *d1_ip = r1_i + (f_i * d_r + f_r * d_i);
    d1_ip += 2;
  }
}

void dsp::Response::integrate (float* data1, float* data2, int ichan)
{
  if (ndim != 1)
    throw Error (InvalidState, "dsp::Response::integrate",
                 "ndim=%u != 1", ndim);

  if (npol != 4)
    throw Error (InvalidState, "dsp::Response::integrate (float*, float*)",
                 "npol=%u != 4", npol);

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  float* data = buffer + ichan * ndat * ndim;

  if (verbose)
    cerr << "dsp::Response::integrate::cross_detect_int" << endl;

  cross_detect_int (npts, data1, data2,
		    data, data + offset,
		    data + 2*offset, data + 3*offset, 1);
}

void dsp::Response::set (const vector<Jones<float> >& response)
{
  if (verbose)
    cerr << "dsp::Response::set" <<endl;

  // one poln, one channel, Jones
  resize (1, 1, response.size(), 8);

  float* f = buffer;

  for (unsigned idat=0; idat<response.size(); idat++)
  {
    // for efficiency, the elements of a Jones matrix Response
    // are ordered as: j00, j10, j11, j01

    for (int j=0; j<2; j++)
    {
      for (int i=0; i<2; i++) 
      {
        complex<double> element = response[idat]( (i+j)%2, j );
        // Re
        *f = element.real();
        f++;
        // Im
        *f = element.imag();
        f++;
      }
    }
  }
}

// ////////////////////////////////////////////////////////////////
//
// dsp::Response::doswap swaps the passband(s)
//
// If 'each_chan' is true, then the nchan units (channels) into which
// the Response is logically divided will be swapped individually
//
void dsp::Response::doswap (unsigned divisions)
{
  if (nchan == 0)
    throw Error (InvalidState, "dsp::Response::swap",
		 "invalid nchan=%d", nchan);

  unsigned half_npts = (ndat * ndim * nchan) / (2 * divisions);

  if (half_npts < 2)
    throw Error (InvalidState, "dsp::Response::swap",
		 "invalid npts=%d (ndat=%u ndim=%u nchan=%u)",
                 half_npts, ndat, ndim, nchan);

#ifdef _DEBUG
  cerr << "dsp::Response::swap"
    " nchan=" << nchan <<
    " ndat=" << ndat <<
    " ndim=" << ndim <<
    " npts=" << half_npts
       << endl;
#endif

  float* ptr1 = 0;
  float* ptr2 = 0;
  float  temp = 0;

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    ptr1 = buffer + offset * ipol;
    ptr2 = ptr1 + half_npts;

    for (unsigned idiv=0; idiv<divisions; idiv++)
    {

      for (unsigned ipt=0; ipt<half_npts; ipt++)
      {
        temp = *ptr1;
        *ptr1 = *ptr2; ptr1++;
        *ptr2 = temp; ptr2++;
      }

      ptr1+=half_npts;
      ptr2+=half_npts;
    }
  }

  if (divisions == 1)
    whole_swapped = !whole_swapped;
  else if (divisions == swap_divisions)
    swap_divisions = 0;
  else
    swap_divisions = divisions;
}

void dsp::Response::flagswap (unsigned divisions)
{
  if (divisions == 1)
    whole_swapped = true;
  else
    swap_divisions = divisions;
}

void dsp::Response::check_finite (const char* name)
{
  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      float* tmp = get_datptr (ichan, ipol);
      for (unsigned idat=0; idat < ndat; idat++)
      {
        for (unsigned idim=0; idim < ndim; idim++)
        {
          if (!true_math::finite(*tmp))
            throw Error (InvalidState, name,
                         "not finite: ichan=%u ipol=%u idat=%u idim=%u", 
                         ichan, ipol, idat, idim);
          tmp ++;
        }
      }
    }
  }
}
