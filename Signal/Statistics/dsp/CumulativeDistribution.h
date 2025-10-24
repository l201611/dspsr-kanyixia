//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 - 2023 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Signal_Statistics_CumulativeDistribution_h
#define __dsp_Signal_Statistics_CumulativeDistribution_h

#include "dsp/ProbabilityDensity.h"
#include "dsp/Romberg.h"
#include "ReferenceTo.h"

namespace dsp {

  /*! Used to compute the cumulative function (CF) and complementary CF (CCF) of a given PDF */
  class CumulativeDistribution : public Reference::Able 
  {

  public:

    //! Construct with optional PDF
    CumulativeDistribution(ProbabilityDensity* pdf = 0) { if (pdf) set_pdf(pdf); }

    //! Set the probability density function
    void set_pdf (ProbabilityDensity* _pdf) { pdf = _pdf; }

    //! Return the probability density at x
    double operator() (double x) { return pdf->evaluate(x); }

    //! Return cumulative function for x
    double cf (double x);
    double log_cf (double x);
    double dlog_cf (double x);

    //! Return complementary cumulative function for 
    double ccf (double x);
    double log_ccf (double x);
    double dlog_ccf (double x);
      
    typedef double argument_type;
    typedef double return_type;

  protected:
  
    Reference::To<ProbabilityDensity> pdf;
    
    // used to integrate the CF and/or CCF
    Romberg<MidPoint> midpoint;
    Romberg<> normal;
  };
}

#endif
