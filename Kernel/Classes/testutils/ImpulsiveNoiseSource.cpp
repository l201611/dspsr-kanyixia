/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <dsp/ImpulsiveNoiseSource.h>

#include <cmath>

dsp::test::ImpulsiveNoiseSource::ImpulsiveNoiseSource(unsigned _niterations) : TestSource("ImpulsiveNoiseSource", _niterations) {}

void dsp::test::ImpulsiveNoiseSource::set_impulse_duration(double _impulse_duration)
{
  if (_impulse_duration < 0.0)
    throw Error (InvalidParam, "dsp::test::ImpulsiveNoiseSource::set_impulse_duration",
		 "invalid impulse_duration=%f must not be negative", _impulse_duration);

  impulse_duration = _impulse_duration;
}

void dsp::test::ImpulsiveNoiseSource::set_period(double _period)
{
  if (_period <= 0.0)
    throw Error (InvalidParam, "dsp::test::ImpulsiveNoiseSource::set_period",
		 "invalid period=%f must be greater than zero", _period);

  period = _period;
}

void dsp::test::ImpulsiveNoiseSource::generate_fpt()
{
  const auto ndat = output->get_ndat();
  const auto nchan = output->get_nchan();
  const auto ndim = output->get_ndim();
  const auto npol = output->get_npol();

  const double duty_cycle = impulse_duration / period;
  const double phase_per_sample = output->get_rate() / period;

  if (verbose)
    std::cerr << "dsp::test::ImpulsiveNoiseSource::generate_fpt - ndat=" << ndat
      << ", nchan=" << nchan << ", ndim=" << ndim << ", npol=" << npol
      << ", iterations=" << iterations << ", output->get_rate()=" << output->get_rate()
      << ", duty_cycle=" << duty_cycle << ", phase_per_sample=" << phase_per_sample
      << ", output_order=" << output_order
      << std::endl;

  for (auto ichan = 0; ichan < nchan; ichan++)
  {
    for (auto ipol = 0; ipol < npol; ipol++)
    {
      float *ptr = output->get_datptr(ichan, ipol);

      size_t ival = 0;
      for (auto idat = 0; idat < ndat; idat++)
      {
        auto frac_phase = fmod(static_cast<double>(current_samples + idat) * phase_per_sample + static_cast<double>(phase_offset), 1.0);
        float samp_value = frac_phase < duty_cycle ? height : 0.0;

        if (verbose)
        {
          if (samp_value > 0.0 && ichan == 0 && ipol == 0)
          {
            std::cerr << "dsp::test::ImpulsiveNoiseSource::generate_fpt - Impulse at idat=" << idat << ", frac_phase=" << frac_phase << std::endl;
          }
        }

        for (auto idim = 0; idim < ndim; idim++, ival++)
        {
          ptr[ival] = samp_value;
        }
      }
    }
  }
}

void dsp::test::ImpulsiveNoiseSource::generate_tfp()
{
  const auto ndat = output->get_ndat();
  const auto nchan = output->get_nchan();
  const auto ndim = output->get_ndim();
  const auto npol = output->get_npol();

  const double duty_cycle = impulse_duration / period;
  const double phase_per_sample = output->get_rate() / period;

  if (verbose)
    std::cerr << "dsp::test::ImpulsiveNoiseSource::generate_tfp - ndat=" << ndat
      << ", nchan=" << nchan << ", ndim=" << ndim << ", npol=" << npol
      << ", iterations=" << iterations << ", output->get_rate()=" << output->get_rate()
      << ", duty_cycle=" << duty_cycle << ", phase_per_sample=" << phase_per_sample
      << ", output_order=" << output_order
      << std::endl;

  float *ptr = output->get_dattfp();
  uint64_t ival = 0;
  for (auto idat = 0; idat < ndat; idat++)
  {
    auto frac_phase = fmod(static_cast<double>(current_samples + idat) * phase_per_sample + static_cast<double>(phase_offset), 1.0);
    float samp_value = frac_phase < duty_cycle ? height : 0.0;

    if (verbose)
    {
      if (samp_value > 0.0)
      {
        std::cerr << "dsp::test::ImpulsiveNoiseSource::generate_tfp - Impulse at idat=" << idat << ", frac_phase=" << frac_phase << std::endl;
      }
    }

    for (auto ichan = 0; ichan < nchan; ichan++)
    {
      for (auto ipol = 0; ipol < npol; ipol++)
      {
        for (auto idim = 0; idim < ndim; idim++, ival++)
        {
          ptr[ival] = samp_value;
        }
      }
    }
  }
}

void dsp::test::ImpulsiveNoiseSource::operation()
{
  switch (output_order)
  {
  case dsp::TimeSeries::OrderFPT:
    generate_fpt();
    break;
  case dsp::TimeSeries::OrderTFP:
  default:
    generate_tfp();
    break;
  }

  // need to know how many samples produced
  current_samples += output->get_ndat();

  iterations++;
  set_end_of_data(iterations >= niterations);

  if (verbose)
    std::cerr << "ImpulsiveNoiseSource::operation() - complete. eod=" << eod << std::endl;
}

dsp::Source* dsp::test::ImpulsiveNoiseSource::clone() const
{
  ImpulsiveNoiseSource * clone = new dsp::test::ImpulsiveNoiseSource();
  clone->set_output(output);
  return clone;
}
