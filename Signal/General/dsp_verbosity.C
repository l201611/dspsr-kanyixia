/***************************************************************************
 *
 *   Copyright (C) 2007 - 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Observation.h"
#include "dsp/Operation.h"
#include "dsp/Shape.h"
#include "dsp/OptimalFFT.h"

// defined in Kernel/Classes/base_verbosity.C
extern unsigned base_verbosity_level;

void dsp::set_verbosity (unsigned level)
{
  base_verbosity_level = level;
  dsp::Observation::verbose = (level >= 3);
  dsp::Operation::verbose =   (level >= 3);
  dsp::Shape::verbose =       (level >= 3);
  dsp::OptimalFFT::verbose =  (level >= 3);
}

