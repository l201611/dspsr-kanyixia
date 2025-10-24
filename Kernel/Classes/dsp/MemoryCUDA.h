//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_MemoryCUDA_h_
#define __dsp_MemoryCUDA_h_

#include "dsp/MemoryHost.h"

#include <cuda_runtime.h>

namespace CUDA
{
  //! Manages CUDA pinned memory allocation and destruction
  class PinnedMemory : public dsp::MemoryHost
  {
  public:
    void* do_allocate (size_t nbytes) override;
    void  do_free (void*) override;
  };

  //! Manages CUDA device memory allocation and destruction
  class DeviceMemory : public dsp::Memory
  {
  public:

    DeviceMemory (cudaStream_t _stream = 0, int _device = 0);

    void* do_allocate (size_t nbytes);
    void do_free (void*);
    void do_copy (void* to, const void* from, size_t bytes);
    void do_zero (void*, size_t);
    bool on_host () const { return false; }

    void set_stream (cudaStream_t _stream) { stream = _stream; }
    cudaStream_t get_stream () { return stream; }
    cudaStream_t get_stream () const { return stream; }

    int get_device () { return device; };
    int get_device () const { return device; };


  protected:
    cudaStream_t stream;
    int device;
  };

  class SharedPinnedMemory : public dsp::Memory
  {
  public:
    void * do_allocate (size_t nbytes);
    void do_free (void*);
  };

}

#endif
