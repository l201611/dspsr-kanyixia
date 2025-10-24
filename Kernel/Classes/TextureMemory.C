/***************************************************************************
 *
 *   Copyright (C) 2012 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MemoryCUDA.h"
#include "debug.h"

#include <cuda_runtime.h>

#include <iostream>
using namespace std;

/***************************************************************************
 *
 *   Texture Memory on device
 *
 ***************************************************************************/

void* CUDA::TextureMemory::do_allocate (size_t nbytes)
{
  if (nbytes == 0)
    cerr << "CUDA::TextureMemory::do_allocate nbytes==0!" << endl;
  DEBUG("CUDA::TextureMemory::do_allocate cudaMalloc (" << nbytes << ")");
  void* ptr = 0;
  cudaError error = cudaMalloc (&ptr, nbytes);
  if (error != cudaSuccess)
  {
    int device;
    cudaGetDevice (&device);
    throw Error (InvalidState, "CUDA::TextureMemory::do_allocate",
                 "cudaMalloc failed on device %d: %s", device, cudaGetErrorString(error));
  }
  DEBUG("CUDA::TextureMemory::allocate cudaMalloc ptr=" << ptr);
  texture_size = nbytes;
  return ptr;
}


void CUDA::TextureMemory::do_zero (void* ptr, size_t nbytes)
{
  DEBUG("CUDA::TextureMemory::do_zero ptr=" << ptr << " nbytes=" << nbytes);
  cudaError_t error;
  if (stream)
    error = cudaMemsetAsync (ptr, 0, nbytes, stream);
  else
    error = cudaMemset (ptr, 0, nbytes);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::TextureMemory::do_zero",
                 "cudaMemset%s (%x, 0, %u): %s", stream?"Async":"", 
                 ptr, nbytes, cudaGetErrorString (error));
}

void CUDA::TextureMemory::do_free (void* ptr)
{
  DEBUG("CUDA::TextureMemory::free cudaFree (" << ptr << ")");
  cudaFree (ptr);
  if (texture_ref)
  {
    DEBUG("CUDA::TextureMemory::free cudaUnbindTexture(" << texture_ref << ")");
    cudaUnbindTexture(texture_ref);
    texture_ref = 0;
  }
}

void CUDA::TextureMemory::do_copy (void* to, const void* from, size_t bytes)
{
  DEBUG("CUDA::TextureMemory::copy (" << to <<","<< from <<","<< bytes << ")");
  cudaError_t error;
  if (stream)
    error = cudaMemcpyAsync (to, from, bytes, cudaMemcpyDeviceToDevice, stream);
  else
    error = cudaMemcpy (to, from, bytes, cudaMemcpyDeviceToDevice);
  if (error != cudaSuccess)
  {
    int device;
    cudaGetDevice (&device);
    throw Error (InvalidState, "CUDA::TextureMemory::do_copy",
                 "cudaMemcpy%s failed on device %d: %s", stream?"Async":"", 
                 device, cudaGetErrorString(error));
  }
}

void CUDA::TextureMemory::set_format_signed (int x, int y, int z, int w)
{
  channel_desc = cudaCreateChannelDesc(x, y, z, w, cudaChannelFormatKindSigned);
}

void CUDA::TextureMemory::set_format_unsigned (int x, int y, int z, int w)
{
  channel_desc = cudaCreateChannelDesc(x, y, z, w, cudaChannelFormatKindUnsigned);
}

void CUDA::TextureMemory::set_format_float (int x, int y, int z, int w)
{
  channel_desc = cudaCreateChannelDesc(x, y, z, w, cudaChannelFormatKindFloat);
}

void CUDA::TextureMemory::activate (const void * ptr)
{
  DEBUG("CUDA::TextureMemory::activate (" << ptr << ")");
  cudaError_t error = cudaBindTexture(0, texture_ref, ptr, &channel_desc, texture_size);
  if (error != cudaSuccess)
  {
    int device;
    cudaGetDevice (&device);
    throw Error (InvalidState, "CUDA::TextureMemory::activate",
                 "cudaBindTexture ref=%p, ptr=%p failed on device %d: %s", 
                 texture_ref, ptr, device, cudaGetErrorString(error));
  }
}

void CUDA::TextureMemory::set_symbol ( const char * symbol)
{
  DEBUG("CUDA::TextureMemory::set_symbol" << symbol << ")");
  cudaError_t error = cudaGetTextureReference (&texture_ref, symbol);
  if (error != cudaSuccess)
  {
    int device;
    cudaGetDevice (&device);
    throw Error (InvalidState, "CUDA::TextureMemory::set_symbol",
                 "cudaGetTextureReference failed on device %d: %s", 
                 device, cudaGetErrorString(error));
  }
  DEBUG("CUDA::TextureMemory::set_symbol texture_ref=" << texture_ref);
}

