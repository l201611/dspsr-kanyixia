/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "dsp/Derotation.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "ThreadContext.h"
#include "Error.h"

#include <complex>

using namespace std;

//! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
void dsp::Derotation::set_rotation_measure (double _rotation_measure)
{
  if (verbose)
    cerr << "dsp::Derotation::set_rotation_measure " << _rotation_measure << endl;

  if (get_rotation_measure() != _rotation_measure)
    changed();

  birefringence.set_rotation_measure (_rotation_measure);
}

double dsp::Derotation::get_rotation_measure () const
{
  return birefringence.get_rotation_measure().get_value();
}

void dsp::Derotation::configure (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Derotation::prepare nchan=" << channels
       << " rm=" << input->get_rotation_measure() << endl;

  npol = 1;
  ndim = 8;
  
  set_rotation_measure ( input->get_rotation_measure() );
  PlasmaResponse::configure ( input, channels );
}

void dsp::Derotation::build_setup (double chan_freq)
{
  birefringence.set_reference_frequency (chan_freq);
}

Jones<float> dsp::Derotation::build_compute (double chan_freq, double freq)
{
  birefringence.set_frequency (chan_freq + freq);
  Jones<float> J = inv( birefringence.evaluate () );

  if (!true_math::finite(J))
    throw Error (InvalidState, "dsp::Derotation::build_compute",
                 "non-finite J="+tostring(J) + 
                 " ref_freq="+tostring(chan_freq) + " freq="+tostring(freq));

  return J;
}

void dsp::Derotation::build (const dsp::Observation*)
{
  vector< Jones<float> > response (ndat * nchan);
  PlasmaResponse::build (response.data(), "rotation", get_rotation_measure(), 
                         this, ndat, nchan);
  
  unsigned _nchan = nchan;
  unsigned _ndat = ndat;
  set (response);
  resize (1, _nchan, _ndat, 8);

  check_finite ("dsp::Derotation::build");

  if (verbose)
    cerr << "dsp::Derotation::build nchan=" << nchan << " nfilt=" << ndat << endl;
}

/*!
  return x squared
  */
template <typename T> inline T sqr (T x) { return x*x; }

double dsp::Derotation::delay_time (double freq) const
{
  Calibration::Faraday temp;

  temp.set_rotation_measure (get_rotation_measure());
  temp.set_reference_wavelength (0.0);
  temp.set_frequency (freq);

  double abs_delta_PA = temp.get_rotation () / (2.0*M_PI);
  double abs_dgd_mus = 2.0 * fabs(abs_delta_PA) / freq;
  return abs_dgd_mus * 1e-6;
}

