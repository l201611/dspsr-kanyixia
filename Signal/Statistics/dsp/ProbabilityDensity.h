//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 - 2023 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Signal_Statistics_ProbabilityDensity_h
#define __dsp_Signal_Statistics_ProbabilityDensity_h

#include "ReferenceAble.h"

namespace dsp {

  class ProbabilityDensity : public Reference::Able 
  {
    public:
      //! Compute the probability density at x
      virtual double evaluate (double x) = 0; 
  };

}

#endif
