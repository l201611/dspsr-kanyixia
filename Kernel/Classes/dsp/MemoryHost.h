//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_MemoryHost_h_
#define __dsp_MemoryHost_h_

#include "Memory.h"

namespace dsp {

  //! Manages memory allocation and destruction on the host machine
  class MemoryHost : public Memory
  {
  public:
    void* do_allocate (size_t nbytes) override;
    void  do_free (void*) override;
    void  do_zero (void* ptr, size_t nbytes) override;
    void  do_copy (void* to, const void* from, size_t bytes) override;
    bool  on_host () const override { return true; }
  };

}

#endif
