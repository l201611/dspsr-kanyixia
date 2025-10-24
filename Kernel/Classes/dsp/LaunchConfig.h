//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2013-2025 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_cuda_LaunchConfig_h
#define __dsp_cuda_LaunchConfig_h

#include <cuda_runtime.h>

namespace CUDA
{
  //! Base class of launch configuration helpers
  class LaunchConfig
  {
  protected:

    //! device that whose properties to query
    int device;

    //! CUDA device property structure
    struct cudaDeviceProp device_properties;

  public:
    //! notes that init has not yet been called by setting device = -1
    LaunchConfig () { device = -1; }

    //! gets the current device ID and calls cudaGetDeviceProperties
    void init ();

    //! return the maximum number of threads per block for the device
    size_t get_max_threads_per_block ();

    //! return the maximum amount of shared memory for the device
    size_t get_max_shm ();

    //! return the maximum supported dimensions of a CUDA grid for the device
    dim3 get_max_grid_size ();

    //! return the maximum supported X dimension of a CUDA grid for the device
    int get_max_grid_size_x () { return get_max_grid_size().x; };

    //! return the maximum supported Y dimension of a CUDA grid for the device
    int get_max_grid_size_y () { return get_max_grid_size().y; };

    //! return the maximum supported Z dimension of a CUDA grid for the device
    int get_max_grid_size_z () { return get_max_grid_size().y; }

  };


  //! Simple one-dimensional launch configuration
  class LaunchConfig1D : public LaunchConfig
  {
    unsigned nblock;
    unsigned nthread;
    unsigned block_dim;

  public:

    LaunchConfig1D () { block_dim = 0; }

    //! Set the block dimension to be used
    /*!
      default: block_dim == 0 and
      element index = blockIdx.x*blockDim.x + threadIdx.x;
    */
    void set_block_dim (unsigned i) { block_dim = i; }

    //! Set the number of elements to be computed
    void set_nelement (unsigned n);

    //! Return the number of blocks into which jobs is divided
    unsigned get_nblock() { return nblock; }

    //! Return the number of threads per block
    unsigned get_nthread() { return nthread; }
  };

} // namespace CUDA

#endif // !defined(__dsp_cuda_LaunchConfig_h)
