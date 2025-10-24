/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ScalarFilter.h"
#include "dsp/Observation.h"

using namespace std;

//! Set the scale factor to be applied by the filter
void dsp::ScalarFilter::set_scale_factor (float _scale_factor)
{
  if (verbose)
    cerr << "dsp::ScalarFilter::set_scale_factor factor=" << _scale_factor << endl;
  if (scale_factor != _scale_factor)
    changed();
  scale_factor = _scale_factor;
}

float dsp::ScalarFilter::get_scale_factor () const
{
  return scale_factor;
}


void dsp::ScalarFilter::build (const Observation*)
{
  // calculate the complex response of the scalar
  complex<float>* phasors = reinterpret_cast< complex<float>* > ( buffer );
  uint64_t npt = ndat * nchan;

  if (verbose)
    cerr << "dsp::ScalarFilter::build scale_factor=" << scale_factor << endl;

  for (unsigned ipt=0; ipt<npt; ipt++)
    phasors[ipt] = complex<float> (scale_factor, 0.0);
}

//! Create an Scalar Filter with nchan channels
void dsp::ScalarFilter::configure (const Observation* obs, unsigned channels)
{
  if (verbose)
    cerr << "dsp::ScalarFilter::configure channels=" << channels << endl;

  if (!channels)
    channels = obs->get_nchan();
  
  if (verbose)
    cerr << "dsp::ScalarFilter::configure set_nchan(" << channels << ")" << endl;

  set_nchan (channels);
}
