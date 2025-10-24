/***************************************************************************
 *
 *   Copyright (C) 2008 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #define _DEBUG 1
#include "debug.h"

#include "dsp/SKLimits.h"
#include "dsp/PearsonIV.h"
#include "dsp/CumulativeDistribution.h"
#include "dsp/NewtonRaphson.h"

#include "Functor.h"
#include "ierf.h"

#include <iostream> 
#include <math.h>

using namespace std;

dsp::SKLimits::SKLimits (unsigned _M, float _std_devs)
{
  M = _M;
  set_std_devs(_std_devs);
}

dsp::SKLimits::~SKLimits ()
{
}

void dsp::SKLimits::set_Nd (unsigned _Nd)
{
  Nd = _Nd;
  lower_threshold = upper_threshold = 0;
}

void dsp::SKLimits::set_M (unsigned _M)
{
  M = _M;
  lower_threshold = upper_threshold = 0;
}

void dsp::SKLimits::set_std_devs (float _std_devs)
{
  family_wise_error_rate = 1.0 - erf((float) _std_devs / sqrt(2));
  lower_threshold = upper_threshold = 0;

#if _DEBUG
  // verify that sigma can be recovered from FWER
  double inv_std_devs = ierf(1.0 - family_wise_error_rate) * sqrt(2);
  cerr << "dsp::SKLimits::set_std_devs sigma=" <<_std_devs << " -> " << inv_std_devs << endl;
#endif
}

void dsp::SKLimits::set_fwer (float _fwer)
{
  family_wise_error_rate = _fwer;
  lower_threshold = upper_threshold = 0;
}

void dsp::SKLimits::set_ntest (unsigned _ntest)
{
  ntest = _ntest;
  lower_threshold = upper_threshold = 0;
}

/*! Calculate the standard statistical parameters for Nyquist-sampled intensities */
dsp::PearsonIV::Parameters dsp::SKLimits::parametersIV (double M)
{
  DEBUG("SKLimits::parametersIV M=" << M);

  PearsonIV::Parameters p;

  p.mu1 = 1;

  p.mu2 = (4 * M * M) / ((M-1) * (M+2) * (M+3));

  p.beta1  = (4 * (M+2) * (M+3) * (5*M - 7) * (5*M - 7));
  p.beta1 /= ((M-1) * (M+4) * (M+4) * (M+5) * (M+5));

  p.beta2 = 3 * (M+2) * (M+3) * (M*M*M + 98*M*M - 185*M + 78);
  p.beta2 /= ((M-1) * (M+4) * (M+5) * (M+6) * (M+7));

  DEBUG("SKLimits::parametersIV mu1=" << p.mu1 << " mu2=" << p.mu2 << " beta1=" << p.beta1 << " beta2=" << p.beta2);

  return p;
}

/*! Calculate the standard statistical parameters for integrated intensities */
dsp::PearsonIV::Parameters dsp::SKLimits::parametersIV (double M, double Nd)
{
  if (Nd == 1.0)
    return parametersIV(M);

  DEBUG("SKLimits::parametersIV M=" << M << " Nd=" << Nd);

  dsp::PearsonVI::Parameters pvi = parametersVI (M, Nd);

  PearsonIV::Parameters piv;
  piv.mu1 = pvi.mu1;
  piv.mu2 = pvi.mu2;
  
  piv.beta1 = pow(pvi.mu3,2) / pow(pvi.mu2,3);
  piv.beta2 = pvi.mu4 / pow(pvi.mu2,2);

  return piv;
}

/*! Calculate the standard statistical parameters for integrated intensities */
dsp::PearsonVI::Parameters dsp::SKLimits::parametersVI (double M, double Nd)
{
  DEBUG("SKLimits::parametersVI M=" << M << " Nd=" << Nd);

  PearsonVI::Parameters p;

  p.mu1 = 1;

  double b = double(M)/double(M-1);
  double log_gam_num = lgamma(M*Nd + 2);

  p.mu2 = 2 * Nd * (Nd+1) * M*b * exp(log_gam_num - lgamma(M*Nd + 4)); 

  p.mu3 = 8 * Nd * (Nd+1) * M*pow(b,2) * exp(log_gam_num - lgamma(M*Nd + 6));
  p.mu3 *= (Nd+4)*M*Nd -5*Nd - 2;

  p.mu4 = 12 * Nd * (Nd+1) * M*pow(b,3) * exp(log_gam_num - lgamma(M*Nd + 8));
  p.mu4 *= (pow(M,3) + 3*pow(M,2))*pow(Nd,4) + (pow(M,3) + 68*pow(M,2) - 93*M)*pow(Nd,3) 
    + (125*pow(M,2) - 245*M + 84)*pow(Nd,2) + (48 - 32*M)*Nd + 24;

  return p;
}

void dsp::SKLimits::calc_limits()
{
  if ((M < 2) || (family_wise_error_rate <= 0))
  {
    throw Error (InvalidParam, "SKLimits::calc_limits",
      "invalid inputs, M=%u fwer=%f", M, family_wise_error_rate);
  }

  per_comparison_error_rate = 1.0 - pow(1 - family_wise_error_rate, 1.0 / ntest);

  DEBUG("dsp::SKLimits::calc_limits fwer=" << family_wise_error_rate << " -> alpha=" << per_comparison_error_rate);

  double target = per_comparison_error_rate / 2.0;

  per_comparison_std_devs = ierf(1.0 - per_comparison_error_rate) * sqrt(2);

  if (M >= 32768)
  {
    double one_std_dev = sqrt(4.0 / (double) M);
    double factor = one_std_dev * per_comparison_std_devs;

    lower_threshold = 1.0f - factor;
    upper_threshold = 1.0f + factor;
    return;
  } 

  auto paramsIV = parametersIV(M,Nd);
  double kappa = paramsIV.get_kappa();
  double mu2 = paramsIV.mu2;

  DEBUG("dsp::SKLimits::calc_limits kappa=" << kappa);

  // see Equation 11 of Nita & Gary (2010b) and the text that follows it
  if (kappa < 0)
  {
    throw Error (InvalidState, "dsp::SKLimits::calc_limits",
      "Pearson Type I (kappa < 0) not implemented");
  }
  else if (kappa > 1)
  {
    DEBUG("dsp::SKLimits::calc_limits kappa > 1 - using Pearson Type VI");
    pdf = new PearsonVI(parametersVI(M,Nd));
  }
  else
  {
    DEBUG("dsp::SKLimits::calc_limits 0 < kappa < 1 - using Pearson Type IV");
    pdf = new PearsonIV(paramsIV);
  }

  dsp::CumulativeDistribution cdf (pdf);

  double one_std_dev = sqrt(mu2);
  symmetric_threshold = one_std_dev * per_comparison_std_devs;

  DEBUG("SKLimits::calc_limits M=" << M << " std_devs=" << std_devs << " percent_std_devs=" << percent_std_devs);
  DEBUG("SKLimits::calc_limits 1 std_dev=" << one_std_dev << " symmetric_threshold=" << symmetric_threshold);
  DEBUG("SKLimits::calc_limits target=" << target);

  NewtonRaphson invert;
  invert.upper_limit = 1 + 2*symmetric_threshold;
  invert.lower_limit = 1 - 2*symmetric_threshold;

  double x_guess = 0;

  try
  {
    x_guess = 1 - symmetric_threshold;
    DEBUG("SKLimits::calc_limits CF x_guess=" << x_guess);
    Functor<double(double)> mycf ( &cdf, &dsp::CumulativeDistribution::log_cf );
    Functor<double(double)> mydcf ( &cdf, &dsp::CumulativeDistribution::dlog_cf );
    lower_threshold = invert(mycf, mydcf, log(target), x_guess);
  }
  catch (Error& error)
  {
    cerr << "SKLimits::calc_limits NewtonRaphson on CF failed" << endl;
    lower_threshold = x_guess;
  }

  try
  {
    x_guess = 1 + symmetric_threshold;
    DEBUG("SKLimits::calc_limits CCF x_guess=" << x_guess);
    Functor<double(double)> myccf ( &cdf, &CumulativeDistribution::log_ccf );
    Functor<double(double)> mydccf ( &cdf, &CumulativeDistribution::dlog_ccf );
    upper_threshold = invert(myccf, mydccf, log(target), x_guess);
  }
  catch (Error& error)
  {
    cerr << "SKLimits::calc_limits NewtonRaphson on CCF failed" << endl;
    upper_threshold = x_guess;
  }
  
  DEBUG("SKLimits::calc_limits [" << lower_threshold << " - " << upper_threshold << "]");
}
