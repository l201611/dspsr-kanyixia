//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 - 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Memory_h_
#define __dsp_Memory_h_

#include "Reference.h"

namespace dsp {

  //! Pure virtual base class of objects that manage memory allocation and destruction
  class Memory : public Reference::Able
  {
  protected:
    static Memory* manager;

  public:
    static void* allocate (size_t nbytes);
    static void free (void*);
    static void set_manager (Memory*);
    static Memory* get_manager ();

    virtual void* do_allocate (size_t nbytes) = 0;
    virtual void  do_free (void*) = 0;
    virtual void  do_zero (void* ptr, size_t nbytes) = 0;
    virtual void  do_copy (void* to, const void* from, size_t bytes) = 0;
    virtual bool  on_host () const = 0;
  };

}

#endif
