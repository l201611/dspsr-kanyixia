/***************************************************************************
 *
 *   Copyright (C) 2002-2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #define _DEBUG 1
#include "debug.h"

#include "config.h"
#include "dsp/DedispersionSampleDelay.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "ThreadContext.h"
#include "Error.h"

#include <complex>

using namespace std;

/*! Although the value:

  \f$ DM\,({\rm pc\,cm^{-3}})=2.410331(2)\times10^{-4}D\,({\rm s\,MHz^{2}}) \f$

  has been derived from "fundamental and primary physical and
  astronomical constants" (section 3 of Backer, Hama, van Hook and
  Foster 1993. ApJ 404, 636-642), the rounded value is in standard
  use by pulsar astronomers (page 129 of Manchester and Taylor 1977).
*/
const double dsp::Dedispersion::dm_dispersion = 2.41e-4;

dsp::Dedispersion::Dedispersion ()
{
  /* 
  This empty constructor allows 
  1. only a forward declaration of the Dedispersion::SampleDelay class; and
  2. a Reference::To<SampleDelay> attribute
  in the declaration of the Dedispersion class in the Dedispersion.h header.
  */
}

dsp::Dedispersion::~Dedispersion ()
{
  // See the note about the empty constructor
}

//! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
void dsp::Dedispersion::set_dispersion_measure (double _dispersion_measure)
{
  if (verbose)
    cerr << "dsp::Dedispersion::set_dispersion_measure dm=" << _dispersion_measure << endl;
 
  if (dispersion_measure != _dispersion_measure)
    changed();

  dispersion_measure = _dispersion_measure;
}

void dsp::Dedispersion::set_sample_delay (SampleDelay* delay)
{
  if (sample_delay && sample_delay != delay)
    changed();

  sample_delay = delay;
}

auto dsp::Dedispersion::get_sample_delay () -> SampleDelay* { return sample_delay; }

void dsp::Dedispersion::configure (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Dedispersion::configure input.nchan=" << input->get_nchan()
	 << " channels=" << channels << "\n\t"
      " centre frequency=" << input->get_centre_frequency() <<
      " bandwidth=" << input->get_bandwidth () <<
      " dispersion measure=" << input->get_dispersion_measure() << endl;

  set_dispersion_measure ( input->get_dispersion_measure() );
  PlasmaResponse::configure ( input, channels );
}

void dsp::Dedispersion::build (const Observation* input)
{
  if (sample_delay && input)
  {
    if (verbose)
      cerr << "dsp::Dedispersion::build configuring Dedispersion::SampleDelay" << endl;

    Reference::To<Observation> resulting = new Observation(*input);

    double chanwidth = fabs(input->get_bandwidth()) / double(nchan);
    resulting->set_rate(chanwidth * 1e6);  // bw in MHz; rate in Hz

    sample_delay->init(resulting);
  }

  if (verbose)
    cerr << "dsp::Dedispersion::build nchan=" << nchan << " nfilt=" << ndat
         << " dm=" << get_dispersion_measure() << endl;

  // calculate the complex frequency response function
  vector<float> phases (ndat * nchan);
  build (phases, ndat, nchan);

  complex<float>* phasors = reinterpret_cast< complex<float>* > ( buffer );
  uint64_t npt = ndat * nchan;

  for (unsigned ipt=0; ipt<npt; ipt++)
    phasors[ipt] = polar (float(1.0), phases[ipt]);

#ifdef _DEBUG
  for (unsigned ipt=0; ipt<npt; ipt++)
    cerr << "Dedispersion::build ipt=" << ipt << " " << buffer[ipt*2] << " " << buffer[ipt*2+1] << endl;
#endif

  if (verbose)
    cerr << "dsp::Dedispersion::build done. nchan=" << nchan << " nfilt=" << ndat << endl;
}

void dsp::Dedispersion::build (std::vector<float>& phases, unsigned _npts, unsigned _nchan)
{
  if (verbose)
    cerr << "dsp::Dedispersion::build std::vector<float> nchan=" << _nchan << " nfilt=" << _npts << endl;

  phases.resize(_npts * _nchan);
  PlasmaResponse::build (phases.data(), "dispersion", dispersion_measure, this, _npts, _nchan);
}

/*!
  return x squared
  */
template <typename T> inline T sqr (T x) { return x*x; }

double dsp::Dedispersion::delay_time (double freq) const
{
  double dispersion = dispersion_measure/dm_dispersion;
  return dispersion * 1.0/sqr(freq);
}

void dsp::Dedispersion::build_setup (double chan_freq)
{
  DEBUG("dsp::Dedispersion::build_setup freq=" << chan_freq);

  if (!sample_delay)
  {
    interchannel_fractional_sample_delay = 0.0;
    return;
  }

  auto samp_delay = sample_delay->get_sample_delay(chan_freq);

  DEBUG("dsp::Dedispersion::build_setup samp_delay=" << samp_delay.second);

  double doppler = get_Doppler_shift();
  double bw = get_bandwidth() / doppler;
  double chanwidth = bw / double(nchan);

  // convert fractional sample delay to time delay in microseconds
  interchannel_fractional_sample_delay = samp_delay.second / chanwidth;

  DEBUG("dsp::Dedispersion::build_setup build_setup_delay=" << interchannel_fractional_sample_delay);
}

double dsp::Dedispersion::build_compute (double chan_freq, double freq)
{
  double bw = get_bandwidth();
  double sign = bw / fabs (bw);

  double dispersion_per_MHz = 1e6 * dispersion_measure / dm_dispersion;

  double coeff = -sign * 2*M_PI * dispersion_per_MHz / sqr(chan_freq);

  // additional phase for fractional-sample dispersive delay between channels
  double delay_phase = -2.0*M_PI * freq * interchannel_fractional_sample_delay;

  DEBUG("dsp::Dedispersion::build_compute delay=" << interchannel_fractional_sample_delay << " phase=" << delay_phase);

  double result = coeff*sqr(freq)/(chan_freq+freq) + delay_phase;

  if (build_delays && freq != 0.0)
    result /= (2.0*M_PI * freq);

  return result;
}

//! Build delays in microseconds instead of phases
void dsp::Dedispersion::set_build_delays (bool delays)
{
  build_delays = delays;
}
