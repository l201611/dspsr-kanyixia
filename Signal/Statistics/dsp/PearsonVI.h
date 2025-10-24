//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 - 2023 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Signal_Statistics_PearsonVI_h
#define __dsp_Signal_Statistics_PearsonVI_h

#include "dsp/ProbabilityDensity.h"

namespace dsp {

  //! Pearson VI probability density function
  class PearsonVI : public ProbabilityDensity {

  public:

    class Parameters
    {
      public:
        double mu1;
        double mu2;
        double mu3;
        double mu4;

      double get_alpha();
      double get_beta();
      double get_sqrt(); // returns the common term (inside the sqrt) in alpha and beta
    };
    
    PearsonVI (const Parameters&);

    //! Compute the PDF(x) of Pearson IV
    double evaluate (double x);

    double get_alpha () { return alpha; }
    double get_beta () { return beta; }
    double get_delta () { return delta; }

  private:

    //! Calculate the first four moments of the distribution and ancilliary parameters
    void prepare();

    //! Standard statistical parameters
    Parameters stats;

    double alpha;
    double beta;
    double delta;

    // natural logarithm of Pearson type VI normalisation factor
    double logk;
  };
}

#endif
