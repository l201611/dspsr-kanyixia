/***************************************************************************
 *
 *   Copyright (C) 2008 - 2023 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CumulativeDistribution.h"

double dsp::CumulativeDistribution::cf (double x)
{
  if (x < -1)
    return midpoint(Minus(OneOver(*this)), 0., -1/x);
  else
    return midpoint(Minus(OneOver(*this)), 0., 1.) + normal(*this, -1., x);
}

double dsp::CumulativeDistribution::ccf (double x)
{
  if (x > 1)
    return midpoint(OneOver(*this), 1/x, 0.);
  else
    return midpoint(OneOver(*this), 1., 0.) + normal(*this, x, 1.);
}

double dsp::CumulativeDistribution::log_cf (double x)
{
  return log(cf(x));
}

double dsp::CumulativeDistribution::dlog_cf (double x)
{
  return (*this)(x) / cf(x);
}

double dsp::CumulativeDistribution::log_ccf (double x)
{
  return log(ccf(x));
}

double dsp::CumulativeDistribution::dlog_ccf (double x)
{
  return -1 * (*this)(x) / ccf(x);
}
