/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Scratch.h"
#include "debug.h"

#include <stdlib.h>

using namespace std;

// default scratch space used by Operations
static dsp::Scratch* the_default_scratch = 0;

static Reference::To<dsp::Scratch> the_default_scratch_reference;

dsp::Scratch* dsp::Scratch::get_default_scratch()
{
  if (!the_default_scratch)
  {
    the_default_scratch = new Scratch;

    // an additional reference to the default scratch to ensure it persists
    the_default_scratch_reference = the_default_scratch;
  }
  return the_default_scratch;
}

dsp::Scratch::Scratch ()
{
  working_space = NULL;
  working_size = 0;
  memory = Memory::get_manager();
}

dsp::Scratch::~Scratch ()
{
  DEBUG("dsp::Scratch::~Scratch space(0)");
  space (0);
}

void dsp::Scratch::set_memory (Memory* m)
{
  memory = m;
}

//! Return pointer to a memory resource shared by operations
void* dsp::Scratch::space (size_t nbytes)
{
  DEBUG("dsp::Scratch::space nbytes=" << nbytes << " current=" << working_size);

  if (!nbytes || working_size < nbytes)
  {
    if (working_space)
      memory->do_free (working_space);
    working_space = nullptr;
    working_size = 0;
  }

  if (!nbytes)
    return 0;

  if (working_space == 0)
  {
    working_space = reinterpret_cast<char*>(memory->do_allocate (nbytes));

    if (!working_space)
      throw Error (BadAllocation, "Scratch::space", "error allocating %d bytes", nbytes);

    working_size = nbytes;
  }

  DEBUG("dsp::Scratch::space return start=" << reinterpret_cast<void*>(working_space)
        << " end=" << reinterpret_cast<void*>(working_space + working_size));


  return working_space;
}
