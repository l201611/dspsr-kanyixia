/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/dsp.h"

unsigned base_verbosity_level = 0;

unsigned dsp::get_verbosity()
{
  return base_verbosity_level;
}
