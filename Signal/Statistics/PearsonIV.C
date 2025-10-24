/***************************************************************************
 *
 *   Copyright (C) 2008 - 2023 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG 1
#include "debug.h"

#include "dsp/PearsonIV.h"
#include "true_math.h"
#include "Error.h"

#include <iostream>

#include <math.h>
#include <float.h>

using namespace std;

dsp::PearsonIV::PearsonIV (const Parameters& _stats)
{
  stats = _stats;
  prepare ();
}

dsp::PearsonIV::~PearsonIV ()
{
}

double dsp::PearsonIV::Parameters::get_kappa()
{
  return beta1 * pow(beta2 + 3, 2) / (4 * (4*beta2 - 3*beta1) * (2*beta2 - 3*beta1 -6));
}

void dsp::PearsonIV::prepare ()
{
  DEBUG("PearsonIV::prepare mu1=" << stats.mu1 << " mu2=" << stats.mu2 << " beta1=" << stats.beta1 << " beta2=" << stats.beta2);

  // r, m, nu, a, and lambda defined in Equation (57) of Nita & Gary (2010a)

  r = (6 * (stats.beta2 - stats.beta1 - 1)) / (2*stats.beta2 - 3*stats.beta1 - 6);
  m = (r+2) / 2;
  nu = -1 * (r * (r-2) * sqrt(stats.beta1));
  nu /= sqrt(16 * (r-1) - stats.beta1 * (r-2) * (r-2));
  a = 0.25 * sqrt(stats.mu2 * ((16 * (r-1)) - (stats.beta1 * (r-2) * (r-2))));
  lamda = stats.mu1 - 0.25 * (r-2) * sqrt(stats.mu2) * sqrt(stats.beta1);

  DEBUG("dsp::PearsonIV::prepare r=" << r << " m=" << m << " nu=" << nu << " a=" << a << " lamda=" << lamda);

  // some quick checks
  if (!true_math::finite(nu))
    throw Error (InvalidParam, "dsp::PearsonIV::prepare", "nu is not finite");

  if (!true_math::finite(a))
    throw Error (InvalidParam, "dsp::PearsonIV::prepare", "a is not finite");

  if (!true_math::finite(lamda))
    throw Error (InvalidParam, "dsp::PearsonIV::prepare", "lamda is not finite");

  if (m <= 0.5)
    throw Error (InvalidParam, "dsp::PearsonIV::prepare", "m <= 0.5");

  // calculate the pearson type IV normalisation for these parameters
  logk = log_normalization();

  if (!true_math::finite(logk))
    throw Error (InvalidParam, "dsp::PearsonIV::prepare", "logk not finite");
}

/*!
  Computes the natural logarithm of the right-hand side of Equation 27
  Yuichi Nagahara, Statistics & Probability Letters 43 (1999) 251-264
  
  This equation is attributed to to Abramowitz and Stegun (1965).

  The loggammar2 function is copied verbatim from the source code of the R package,
  PearsonDS: Pearson Distribution System https://CRAN.R-project.org/package=PearsonDS

  The R source code contains the following note:

    Code (partially) adapted from
    CDF/MEMO/STATISTICS/PUBLIC/6820
    A Guide to the Pearson Type IV Distribution
    Joel Heinrich - University of Pennsylvania
    December 21, 2004
*/

double loggammar2(double x,double y) {
  /* returns log(abs(gamma(x+iy)/gamma(x))^2) */
  const double y2=y*y, xmin = (2*y2>10.0) ? 2*y2 : 10.0;
  double r=0, s=1, p=1, f=0;
  while(x<xmin) {
    const double t = y/x++;
    r += log(1 + t*t);
  }
  while (p > s*DBL_EPSILON) {
    p *= y2 + f*f;
    p /= x++ * ++f;
    s += p;
  }
  return -r-log(s);
}

double dsp::PearsonIV::log_nag99_e27()
{
  return loggammar2(m,0.5*nu);
}

/*!
  Computes the natural logarithm of the general normalizing constant defined by 
  Equation 3 of Nagahara (1999).  Note that
  
  Beta(m-0.5,0.5) = Gamma(m-0.5) Gamma(0.5) / Gamma(m)

  and Gamma(0.5) = sqrt(pi)
*/
double dsp::PearsonIV::log_normalization ()
{
  double log_beta = lgamma(m-0.5) + 0.5*log(M_PI) - lgamma(m);
  return log_nag99_e27() - log(a) - log_beta; 
}

double dsp::PearsonIV::evaluate (double x)
{
  double a1 = (x-lamda) / a;

  double log2 = -m * log(1 + a1*a1);
  double log3 = -nu * atan (a1);
  
  double res = exp( logk + log2 + log3 );

  return res;
}
