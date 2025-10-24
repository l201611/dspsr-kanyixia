/***************************************************************************
 *
 *   Copyright (C) 2006 - 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #define _DEBUG 1
#include "debug.h"

#include "dsp/DedispersionSampleDelay.h"
#include "dsp/Observation.h"

using namespace std;

dsp::Dedispersion::SampleDelay::SampleDelay ()
{
  centre_frequency = 0.0;
  bandwidth = 0.0;
  dispersion_measure = 0.0;
  sampling_rate = 0.0;
}

#define SQR(x) (x*x)

bool dsp::Dedispersion::SampleDelay::match (const Observation* obs)
{
  if (verbose)
    std::cerr << "dsp::Dedispersion::SampleDelay::match" << endl;

  bool changed =
    dispersion_measure != obs->get_dispersion_measure() ||
    centre_frequency   != obs->get_centre_frequency() ||
    bandwidth          != obs->get_bandwidth() ||
    sampling_rate      != obs->get_rate() ||
    delays.size()      != obs->get_nchan();

  if (!changed)
    return false;

  init(obs);

  bool error = (sampling_rate == 0 || bandwidth == 0 || centre_frequency == 0);

  if (verbose || error)
    std::cerr << "dsp::Dedispersion::SampleDelay::match"
	      << "\n  centre frequency = " << centre_frequency
	      << "\n  bandwidth = " << bandwidth
	      << "\n  dispersion measure = " << dispersion_measure
	      << "\n  sampling rate = " << sampling_rate << endl;

  if (error)
    throw Error (InvalidParam, "dsp::Dedispersion::SampleDelay::match", "invalid input");

  const unsigned nchan = obs->get_nchan();

  delays.resize(nchan);
  fractional_delays.resize(nchan);
  freqs.resize(nchan);

  for (unsigned ichan = 0; ichan < nchan; ichan++)
  {
    freqs[ichan] = obs->get_centre_frequency (ichan);
    
    auto delay = get_sample_delay(freqs[ichan]);

    delays[ichan] = delay.first;
    fractional_delays[ichan] = delay.second;
  }

  return true;
}

void dsp::Dedispersion::SampleDelay::init (const Observation* obs)
{
  centre_frequency = obs->get_centre_frequency();
  bandwidth = obs->get_bandwidth();
  dispersion_measure = obs->get_dispersion_measure();
  sampling_rate = obs->get_rate();
}

std::pair<int64_t, double> dsp::Dedispersion::SampleDelay::get_sample_delay (double frequency)
{
  // when divided by MHz, yields a dimensionless value
  double dispersion = dispersion_measure / dm_dispersion;

  // Compute the dispersive delay in seconds
  double time_delay = dispersion * (1.0/SQR(centre_frequency) - 1.0/SQR(frequency));
  double sample_delay = time_delay * sampling_rate;

  std::pair<int64_t,double> result;
  result.first = floor(sample_delay + 0.5);
  result.second = sample_delay - double(result.first);

  DEBUG(" freq=" << frequency << " delay=" << time_delay*1e3 << " ms = " << result.first << " + " << result.second << " also " << sample_delay <<  " samples");

  return result;
}

std::pair<double, double> dsp::Dedispersion::SampleDelay::get_time_delay (double frequency)
{
  auto sample_delay = get_sample_delay(frequency);
  std::pair<double, double> result;
  result.first = sample_delay.first/sampling_rate;
  result.second = sample_delay.second/sampling_rate;
  return result;
}

//! Return the dispersion delay for the given frequency channel
int64_t dsp::Dedispersion::SampleDelay::get_delay (unsigned ichan, unsigned ipol) const
{
  if (ichan >= delays.size())
    throw Error (InvalidState, "dsp::Dedispersion::SampleDelay::get_delay", "ichan=%u >= nchan=%u", ichan, delays.size());
  return delays[ichan];
}

double dsp::Dedispersion::SampleDelay::get_fractional_delay (unsigned ichan, unsigned ipol) const
{
  if (ichan >= fractional_delays.size())
    throw Error (
      InvalidState, "dsp::Dedispersion::SampleDelay::get_fractional_delay",
      "ichan=%u >= nchan=%u", ichan, fractional_delays.size()
      );

  return fractional_delays[ichan];
}


//! Return the deispersion delay for the centre of the frequency channel range
int64_t dsp::Dedispersion::SampleDelay::get_delay_range (unsigned schan, unsigned echan, unsigned ipol) const
{
  if (schan > freqs.size() || echan > freqs.size())
    throw Error (InvalidParam, "dsp::Dedispersion::SampleDelay::get_delay_range",
                 "channel range invalid");

  double dispersion = dispersion_measure / dm_dispersion;
  double freq = (freqs[schan] + freqs[echan]) / 2;
  double delay = dispersion * (1.0/SQR(centre_frequency) - 1.0/SQR(freq));

  return int64_t( floor(delay*sampling_rate + 0.5) );
}

void dsp::Dedispersion::SampleDelay::mark (Observation* observation)
{
  observation->set_dispersion_measure (dispersion_measure);
}

