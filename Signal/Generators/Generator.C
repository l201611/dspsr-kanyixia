/**************************************************************************2
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Generator.h"
#include "dsp/MemoryHost.h"

/*! By default, a Generator operates only on host (not device) memory */
bool dsp::Generator::get_device_supported (Memory* memory) const
{
  auto host = dynamic_cast<MemoryHost*>(memory);
  return host != nullptr;
}
