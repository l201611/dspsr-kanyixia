//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 - 2023 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Signal_Statistics_SKLimits_h
#define __dsp_Signal_Statistics_SKLimits_h

#include "dsp/PearsonIV.h"
#include "dsp/PearsonVI.h"
#include "ReferenceTo.h"

namespace dsp {

  //! Computes the upper and lower bounds on estimates of the generalized spectral kurtosis
  /*!
    The bounds are computed by using the Newton-Raphson method to invert the cumulative 
    distribution function (CF) and complementary CF (CCF) of the generalized spectral kurtosis 
    (SK) estimator.  Both the CF and CCF are computed by numerically integrating the probability
    density function (PDF) of SK.  As described by Nita & Gary (2010b; MNRAS 406:60) 
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.406L..60N/abstract, the PDF of SK is approximated
    using either a PearsonIV or PearsonVI distribution.  The decision to use one or the other is
    based on Equation 11 of Nita & Gary (2010b) and the text that follows this equation
    (note that there is a small error in this text, where "Type III" should read "Type VI", 
    as in Figure 1 and its caption).
  */
  class SKLimits {

  public:

    SKLimits (unsigned M, float std_devs);

    ~SKLimits ();

    void calc_limits ();

    double get_lower_threshold() const { return lower_threshold; }

    double get_upper_threshold() const { return upper_threshold; }

    double get_symmetric_threshold() const { return symmetric_threshold; }

    //! Set the number of intensity samples used to estimate the spectral kurtosis
    void set_M (unsigned M);

    //! Set the number of Nyquist-sampled intensities integrated in each intensity sample
    /*! Should be set to ndat * npol * ndim */
    void set_Nd (unsigned Nd);

    //! Set the number of standard deviations used to define the family-wise error rate
    void set_std_devs (float std_devs);

    //! Set the family-wise error rate
    /*! e.g. see https://en.wikipedia.org/wiki/Multiple_comparisons_problem */
    void set_fwer (float fwer);

    //! Set the number of times that a given sample will be tested
    /*! e.g. the number of polarizations times the overlap factor */
    void set_ntest (unsigned ntest);

    //! Get the probability density function appropriate to the input values of M and Nd
    dsp::ProbabilityDensity* get_pdf () { return pdf; }

    //! Equation (53) of Nita & Gary (2010a)
    static PearsonIV::Parameters parametersIV (double M);

    //! Equation (9) of Nita & Gary (2010b) followed by computation of beta1 and beta2
    static PearsonIV::Parameters parametersIV (double M, double Nd);

    //! Equation (9) of Nita & Gary (2010b)
    static PearsonVI::Parameters parametersVI (double M, double Nd);

  private:

    //! Calculate the first four moments of the distribution and ancilliary parameters
    void prepare();

    double lower_threshold = 0.0;

    double upper_threshold = 0.0;

    double symmetric_threshold = 0.0;

    //! Number of intensity samples used to estimate the SK Statisic
    unsigned M = 0;

    //! Number of Nyquist-sampled intensities integrated in each intensity sample
    unsigned Nd = 1;

    //! The family-wise error rate
    double family_wise_error_rate = 0;

    //! The per-comparison error rate
    double per_comparison_error_rate = 0;

    //! The per-comparison effective standard deviations
    double per_comparison_std_devs = 0;

    //! Number of times that a given sample will be tested
    unsigned ntest = 1;

    Reference::To<dsp::ProbabilityDensity> pdf;
  };
}

#endif
