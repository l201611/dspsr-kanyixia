/***************************************************************************
 *
 *   Copyright (C) 2008 - 2023 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG 1
#include "debug.h"

#include "dsp/PearsonVI.h"
#include "true_math.h"
#include "Error.h"

#include <iostream>

#include <math.h>
#include <float.h>

using namespace std;

//! alpha defined in Equation (14) of Nita & Gary (2010b)
double dsp::PearsonVI::Parameters::get_alpha()
{
  return pow(mu3,-3) *
  (
    32*pow(mu2,5) - 4*mu3*pow(mu2,3) + 8*pow(mu3,2)*pow(mu2,2) + pow(mu3,2)*mu2 - pow(mu3,3)
    + ( 8*pow(mu2,3) - mu3*mu2 + pow(mu3,2) ) * get_sqrt() 
  );
}

double dsp::PearsonVI::Parameters::get_sqrt()
{
  return sqrt(16*pow(mu2,4) + (4*mu2 + 1) * mu3*mu3);
}

//! beta defined in Equation (14) of Nita & Gary (2010b)
double dsp::PearsonVI::Parameters::get_beta()
{
  return 3 + 2*mu2*pow(mu3,-2) * ( 4*pow(mu2,2) + get_sqrt() );
}

dsp::PearsonVI::PearsonVI (const Parameters& _stats)
{
  stats = _stats;
  prepare ();
}

void dsp::PearsonVI::prepare ()
{
  DEBUG("PearsonVI::prepare mu1=" << stats.mu1 << " mu2=" << stats.mu2 << " mu3=" << stats.mu3 << " mu4=" << stats.mu4);

  // alpha, beta, and delta defined in Equation (14) and subsequent text of Nita & Gary (2010b)

  alpha = stats.get_alpha();
  beta = stats.get_beta();
  delta = (beta - alpha - 1) / (beta - 1);

  DEBUG("dsp::PearsonVI::prepare alpha=" << alpha << " beta=" << beta << " delta=" << delta);

  // some quick checks
  if (!true_math::finite(alpha))
    throw Error (InvalidParam, "dsp::PearsonVI::prepare", "alpha is not finite");

  if (!true_math::finite(beta))
    throw Error (InvalidParam, "dsp::PearsonVI::prepare", "beta is not finite");

  if (!true_math::finite(delta))
    throw Error (InvalidParam, "dsp::PearsonVI::prepare", "delta is not finite");

  // calculate the pearson type VI normalisation for these parameters
  logk = lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta);

  if (!true_math::finite(logk))
    throw Error (InvalidParam, "dsp::PearsonVI::prepare", "logk not finite");
}

double dsp::PearsonVI::evaluate (double x)
{
  x -= delta;

  if (x >= 0)
    return exp( logk + (alpha-1)*log(x) - (alpha+beta)*log(1+x) );
  else
    return 0;
}

