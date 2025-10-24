/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MemoryHost.h"
#include "malloc16.h"
#include "debug.h"

#include <assert.h>
#include <string.h>

void* dsp::MemoryHost::do_allocate (size_t nbytes)
{
  DEBUG("dsp::MemoryHost::do_allocate (" << nbytes << ")");
  return malloc16 (nbytes);
}

void dsp::MemoryHost::do_free (void* ptr)
{
  DEBUG("dsp::MemoryHost::do_free (" << ptr << ")");
  free16 (ptr);
}

void dsp::MemoryHost::do_zero (void* ptr, size_t nbytes)
{
  DEBUG("dsp::MemoryHost::do_zero (" << (void*) ptr << "," << nbytes << ")");
  memset (ptr, 0, nbytes);
}

void dsp::MemoryHost::do_copy (void* to, const void* from, size_t bytes)
{
  DEBUG("dsp::MemoryHost::copy (" << to << "," << from << "," << bytes << ")");
  memcpy (to, from, bytes);
}
