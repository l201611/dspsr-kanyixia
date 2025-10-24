/***************************************************************************
 *
 *   Copyright (C) 2019 - 2025 by Dean Shaff, Andrew Jameson, and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbankResponse.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"
#include "FTransform.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>

// #define _DEBUG 1

using namespace std;

void dsp::InverseFilterbankResponse::build (const Observation* input)
{
  if (verbose)
  {
    cerr << "dsp::InverseFilterbankResponse::build" << endl;
  }

  if (verbose)
  {
    cerr << "dsp::InverseFilterbankResponse::build input_overlap=" << input_overlap << endl;
    cerr << "dsp::InverseFilterbankResponse::build apply_deripple=" << apply_deripple << endl;
    cerr << "dsp::InverseFilterbankResponse::build impulse_pos=" << impulse_pos << endl;
    cerr << "dsp::InverseFilterbankResponse::build impulse_neg=" << impulse_neg << endl;
    cerr << "dsp::InverseFilterbankResponse::build input_nchan=" << input_nchan
         << " nchan=" << nchan
         << " ndat=" << ndat
         << " npol=" << npol
         << " ndim=" << ndim
         << endl;
  }

  if (! apply_deripple)
  {
    if (ndim != 2)
      throw Error(InvalidState, "dsp::InverseFilterbankResponse::build", "ndim=%d != 2", ndim);

    std::complex<float>* phasors = reinterpret_cast< std::complex<float>* > ( buffer );
    uint64_t npt = ndat * nchan;

    if (npt*ndim > bufsize)
      throw Error(InvalidState, "dsp::InverseFilterbankResponse::build", "bufsize=%d < npt=%d (nchan=%d ndat=%d)", bufsize, npt, nchan, ndat);

    if (verbose)
      cerr << "dsp::InverseFilterbankResponse::build not apply_deripple npt=" << npt << endl;

    for (unsigned ipt=0; ipt<npt; ipt++)
    {
      phasors[ipt] = std::complex<float> (1.0, 0.0);
    }

    if (verbose)
      cerr << "dsp::InverseFilterbankResponse::build not apply_deripple done" << endl;

    return;
  }

  if (verbose)
  {
    cerr << "dsp::InverseFilterbankResponse::build bufsize=" << bufsize << endl;
    cerr << "dsp::InverseFilterbankResponse::build whole_swapped=" << whole_swapped << endl;
    cerr << "dsp::InverseFilterbankResponse::build swap_divisions=" << swap_divisions << endl;
    cerr << "dsp::InverseFilterbankResponse::build pfb_dc_chan=" << pfb_dc_chan << endl;
  }
  unsigned half_chan_shift = 0 ;
  if (pfb_dc_chan)
  {
    half_chan_shift = 1;
  }

  unsigned total_ndat = ndat * nchan;

  std::vector<float> freq_response;
  unsigned ndat_per_chan = total_ndat / input_nchan;
  // calc_freq_response(freq_response, total_ndat*nchan/2);
  calc_freq_response(freq_response, total_ndat/2);
  // roll the array by the appropriate number of bins
  int shift_bins = -1*static_cast<int>(ndim*half_chan_shift*ndat_per_chan/2);
  if (verbose)
  {
    cerr << "dsp::InverseFilterbankResponse::build shift_bins=" << shift_bins << endl;
  }

  std::complex<float>* freq_response_complex = reinterpret_cast<std::complex<float>*>(freq_response.data());

  std::complex<float>* phasors = reinterpret_cast<std::complex<float>*>(buffer);

  uint64_t npt = ndat_per_chan/2;

  int step = 0;
  for (unsigned ichan=0; ichan < input_nchan; ichan++)
  {
    for (uint64_t ipt=0; ipt < npt; ipt++)
    {
      phasors[ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[ipt]), 0.0);
      phasors[npt + ipt + step] = std::complex<float>(1.0/std::abs(freq_response_complex[npt - ipt]), 0.0);
    }
    step += ndat_per_chan;
  }

  if (shift_bins != 0)
  {
    if (verbose)
    {
      cerr << "dsp::InverseFilterbankResponse::build rolling data" << endl;
    }
    int shift_bins_pos =  static_cast<int>(ndat) + shift_bins/static_cast<int>(ndim);
    roll<std::complex<float>>(phasors, ndat, shift_bins_pos);
  }
  if (verbose)
  {
    cerr << "dsp::InverseFilterbankResponse::build done" << endl;
  }
}

void dsp::InverseFilterbankResponse::calc_freq_response (std::vector<float>& freq_response, unsigned n_freq)
{
  if (verbose)
  {
    cerr << "dsp::InverseFilterbankResponse::calc_freq_response:"
      << " freq_response.size()=" << freq_response.size()
      << " n_freq=" << n_freq
      << endl;
  }

  freq_response.resize(2*n_freq);
  std::vector<float> filter_coeff_padded (2*n_freq);
  std::vector<float> freq_response_temp (2*(n_freq+1));
  std::fill(filter_coeff_padded.begin(), filter_coeff_padded.end(), 0.0);

  for (unsigned i=0; i<fir_filter.get_ntaps(); i++)
  {
    filter_coeff_padded[i] = fir_filter[i];
  }

#if _DEBUG
  {
    FILE* fptr = fopen ("fir_filter.dat", "w");
    for (unsigned i=0; i<fir_filter.get_ntaps(); i++)
    {
      fprintf (fptr, "%i %f\n", i, fir_filter[i]);
    }
    fclose (fptr);
  }
#endif

  // need the factor of two for real ("Nyquist") input signal
  forward = FTransform::Agent::current->get_plan (2*n_freq, FTransform::frc);

  forward->frc1d(
    2*n_freq,
    freq_response_temp.data(),
    filter_coeff_padded.data()
  );

  for (unsigned i=0; i<2*n_freq; i++)
  {
    freq_response[i] = freq_response_temp[i];
  }

#if _DEBUG
  {
    std::complex<float>* phasors = reinterpret_cast<std::complex<float>*>(freq_response.data());

    FILE* fptr = fopen ("ripple.dat", "w");
    for (unsigned i=0; i<n_freq; i++)
    {
      auto amp = std::abs(phasors[i]);
      fprintf (fptr, "%i %f\n", i, amp);
    }
    fclose (fptr);
  }
#endif
}

//! Create an Scalar Filter with nchan channels
void dsp::InverseFilterbankResponse::configure (const Observation* obs, unsigned output_nchan)
{
  if (verbose)
  {
    cerr << "dsp::InverseFilterbankResponse::configure output_nchan=" << output_nchan << endl;
  }

  // the sampling rate will be divided by the oversampling_factor
  downsampling = true;

  set_nchan (output_nchan);

  if (verbose)
  {
    cerr << "dsp::InverseFilterbankResponse::configure obs->get_nchan() " << obs->get_nchan() << endl;
    cerr << "dsp::InverseFilterbankResponse::configure ndat=" << ndat << endl;
    cerr << "dsp::InverseFilterbankResponse::configure input_overlap=" << input_overlap << endl;
  }

  input_nchan = obs->get_nchan();
  oversampling_factor = obs->get_oversampling_factor();

  /* The input_overlap attribute is set to the number of samples by which the small forward FFTs in the oversampled
    channelized input data overlap.  This value is
    
    1. divided by 2 because an equal number of samples are discarded from the beginning and end of each output FFT, 
    2. multiplied by input_nchan / output_nchan to reflect the increase in sampling rate and
    3. divided by oversampling_factor to reflect the decrease in sampling rate;
  */

  // input_to_output returns (input_overlap/2) * (input_nchan/output_nchan) / oversampling_factor
  impulse_pos = input_to_output(input_overlap/2, input_nchan, output_nchan, oversampling_factor);
  impulse_neg = input_to_output(input_overlap/2, input_nchan, output_nchan, oversampling_factor);
}
