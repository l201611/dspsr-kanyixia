//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 - 2023 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Signal_Statistics_PearsonIV_h
#define __dsp_Signal_Statistics_PearsonIV_h

#include "dsp/ProbabilityDensity.h"

namespace dsp {

  class PearsonIV : public ProbabilityDensity {

  public:

    class Parameters
    {
      public:
        double mu1;
        double mu2;
        double beta1;
        double beta2;

        //! Equation 11 of Nita & Gary (2010b)
        double get_kappa();
    };
    
    PearsonIV (const Parameters&);
    ~PearsonIV ();

    //! Compute the PDF(x) of Pearson IV
    double evaluate (double x);

    double get_m () { return m; }
    double get_nu () { return nu; }
    double get_lamda () { return lamda; }
    double get_a () { return a; }

    //! return the natural logarithm of Equation 3 of Nagahara (1999)
    double log_normalization();

    //! return the natural logarithm of Equation 27 of Nagahara (1999)
    double log_nag99_e27();

  private:

    //! Calculate the first four moments of the distribution and ancilliary parameters
    void prepare();

    //! Standard statistical parameters
    Parameters stats;

    double r;
    double m;
    double nu;
    double a;
    double lamda;

    // natural logarithm of Pearson type IV normalisation factor
    double logk;
  };
}

#endif
