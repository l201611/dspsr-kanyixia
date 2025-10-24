/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TestResponse.h"
#include "dsp/Observation.h"
#include "dsp/Apodization.h"

#include "FTransform.h"
#include <gtest/gtest.h>

namespace dsp::test
{

void TestResponse::configure (const Observation* input, unsigned channels)
{
  if (impulse_neg == 0)
    throw Error (InvalidState, "TestResponse::configure", "impulse_neg == 0");

  if (impulse_pos == 0)
    throw Error (InvalidState, "TestResponse::configure", "impulse_pos == 0");

  if (ndim != 2)
    throw Error (InvalidState, "TestResponse::configure", "ndim != 2");
}

void create_response(
  const Apodization& amplitude,
  std::vector<std::complex<float>>& temporal,
  std::vector<std::complex<float>>& spectral)
{
  unsigned ndat = spectral.size();
  unsigned size = amplitude.get_ndat();
  if (size > ndat)
    throw Error(InvalidState, "TestResponse::create_response", "ampltiude size=%u > spectral response ndat=%u", size, ndat);

  if (amplitude.get_ndim() != 1)
    throw Error(InvalidState, "TestResponse::create_response", "ampltiude ndim != 1");

  unsigned impulse_pos = amplitude.get_transition_end();
  unsigned impulse_neg = amplitude.get_transition_start();

  if (size != impulse_pos + impulse_neg)
    throw Error(InvalidState, "TestResponse::create_response", "ampltiude size=%u != impulse neg+pos=%u+%u", size, impulse_neg, impulse_pos);

  if (dsp::Shape::verbose)
    std::cerr << "TestResponse::create_response impulse neg=" << impulse_neg << " pos=" << impulse_pos << " ndat=" << ndat << std::endl;

  temporal.resize(size);
  auto amp = amplitude.get_datptr(0,0);

  std::complex<double> sum_temporal = 0.0;

  // the half cycle of a sine wave used to ensure that the mean of the impulse response is zero
  std::vector<float> correction (size);
  double sum_correction = 0.0;

  for (unsigned idat = 0; idat < size; idat++)
  {
    double phase = 2.0 * M_PI * double(idat) / double(size);
    temporal[idat] = amp[idat] * std::complex<float>(cos(phase), sin(phase));
    sum_temporal += temporal[idat];
    correction[idat] = sin(0.5*phase);
    sum_correction += correction[idat];
  }

  auto mean_offset = sum_temporal / sum_correction;

  // ensure that the mean of the impulse response is zero
  for (unsigned idat = 0; idat < size; idat++)
  {
    temporal[idat] -= mean_offset * double(correction[idat]);
  }

  // the temporal response padded with zeros before the fwd FFT
  std::vector<std::complex<float>> padded (ndat, 0.0);

  double total_power_impulse = 0.0;
  std::complex<double> mean_impulse = 0.0;

  for (unsigned idat = 0; idat < impulse_pos; idat++)
  {
    auto val = temporal[idat + impulse_neg];
    padded[idat] = val;
    total_power_impulse += std::norm(val);
    mean_impulse += val;
  }

  unsigned offset = ndat-impulse_neg;

  for (unsigned idat = 0; idat < impulse_neg; idat++)
  {
    auto val = temporal[idat];
    padded[idat + offset] = val;
    total_power_impulse += std::norm(val);
    mean_impulse += val;
  }

  mean_impulse /= ndat;

  auto input = reinterpret_cast<float*>(padded.data());
  auto output = reinterpret_cast<float*>(spectral.data());
  FTransform::fcc1d (ndat, output, input);

  if (FTransform::get_norm() == FTransform::unnormalized)
  {
    float scalefac = 1.0/sqrt(ndat);
    for (unsigned idat = 0; idat < ndat; idat++)
    {
      spectral[idat] *= scalefac;
    }
  }

  double total_power_freq = 0.0;
  for (unsigned idat = 0; idat < ndat; idat++)
  {
    auto val = spectral[idat];
    total_power_freq += std::norm(val);
  }

  double ratio = total_power_freq / total_power_impulse;

  // expect ratio to be 1
  if (dsp::Shape::verbose)
    std::cerr << "TestResponse::create_response mean=" << mean_impulse << " power time=" << total_power_impulse << " freq=" << total_power_freq << " ratio=" << ratio << std::endl;
}

//! Build the response function
void TestResponse::build (const Observation* input)
{
  // double-check configuration
  configure (input);

  //! The window that defines the amplitude of the impulse response
  Apodization amplitude;

  // Create a pulse shaped like a (possibly asymmetric) Tukey window
  amplitude.set_transition_start (impulse_neg);
  amplitude.set_transition_end (impulse_pos);

  unsigned size = impulse_neg+impulse_pos;
  amplitude.set_size(size);
  amplitude.Tukey();

  spectral.resize(ndat);

  create_response(amplitude,temporal,spectral);

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      auto dat = reinterpret_cast<std::complex<float>*>(get_datptr(ichan, ipol));
      for (unsigned idat=0; idat < ndat; idat++)
        dat[idat] = spectral[idat];
    }
  }

  // create_response performs a complex-to-complex forward FFT
  // this stops Response::naturalize/swap_as_needed from swapping the halves
  whole_swapped = (input->get_state() == Signal::Analytic);
}

} // namespace dsp::test
