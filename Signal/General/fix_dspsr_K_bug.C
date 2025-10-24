/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "Pulsar/Application.h"
#include "Pulsar/UnloadOptions.h"
#include "Pulsar/Archive.h"
#include "Pulsar/Integration.h"
#include "Pulsar/Profile.h"

#include "dsp/DedispersionSampleDelay.h"
#include "dsp/Observation.h"

using namespace std;

//
//! Fixes archives folded with buggy dspsr -K
/*! See https://sourceforge.net/p/dspsr/bugs/104 */
//
class bug_fix : public Pulsar::Application
{
public:

  //! Default constructor
  bug_fix ();

  //! Process the given archive
  void process (Pulsar::Archive*);

  void add_options (CommandLine::Menu&) { /* none */ }
};


bug_fix::bug_fix ()
  : Application ("fix_dspsr-K_bug", "fixes dspsr -K bug: https://sourceforge.net/p/dspsr/bugs/104")
{
  add( new Pulsar::UnloadOptions );
}

template <typename T> inline T sqr (T x) { return x*x; }

void bug_fix::process (Pulsar::Archive* archive)
{
  unsigned nsub = archive->get_nsubint();
  unsigned nchan = archive->get_nchan();
  unsigned npol = archive->get_npol();

  double dispersion_measure = archive->get_dispersion_measure();
  double centrefreq = archive->get_centre_frequency();
  double bw = archive->get_bandwidth();
  double chanwidth = bw / nchan;

  dsp::Observation obs;
  obs.set_centre_frequency(centrefreq);
  obs.set_bandwidth(bw);
  obs.set_dispersion_measure(dispersion_measure);
  obs.set_rate(fabs(chanwidth)*1e6);

  dsp::Dedispersion::SampleDelay sample_delay;
  sample_delay.init(&obs);

  double highest_freq = centrefreq + 0.5*fabs(bw-chanwidth);

  // when divided by MHz, yields a dimensionless value
  double dispersion_per_MHz = 1e6 * dispersion_measure / dsp::Dedispersion::dm_dispersion;

  double max_old = 0;
  double min_old = 0;

  double max_new = 0;
  double min_new = 0;

  Pulsar::Integration* subint = archive->get_Integration(0);

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    // Compute the DM delay in microseconds; when multiplied by the
    // frequency in MHz, the powers of ten cancel each other
    double chan_cfreq = subint->get_centre_frequency(ichan);

    double delay_us = dispersion_per_MHz * ( 1.0/sqr(chan_cfreq) - 1.0/sqr(highest_freq) );
    double samp_int = 1.0/chanwidth;
    double old_delay = - fmod(delay_us, samp_int);

    auto samp_delay = sample_delay.get_sample_delay(chan_cfreq);

    // convert fractional sample delay to -ve delay in microseconds
    double new_delay = samp_delay.second / chanwidth;

    min_old = std::min (old_delay, min_old);
    max_old = std::max (old_delay, max_old);

    min_new = std::min (new_delay, min_new);
    max_new = std::max (new_delay, max_new);

    for (unsigned isub=0; isub < nsub; isub++)
    {
      Pulsar::Integration* subint = archive->get_Integration(isub);
      double period = subint->get_folding_period();
      double phase = (old_delay - new_delay) * 1e-6 / period;

      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        Pulsar::Profile* profile = archive->get_Profile (isub, ipol, ichan);
        profile->rotate_phase(phase);
      }
    }
  }
}

int main (int argc, char** argv)
{
  bug_fix program;
  return program.main (argc, argv);
}

