//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019-2025 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>

#include "dsp/UWBFloatUnpackerCUDA.h"
#include "dsp/Operation.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"

using namespace std;

void check_error_stream (const char*, cudaStream_t);

//! unpacks npol=2 inputs from float4 (Re,Im,Re,Im)
__global__ void uwb_unpack_tp_fpt_float_kernel(
  float2 * to,
  const float4 * from,
  uint64_t ndat,
  uint64_t out_pol_stride
)
{
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idat >= ndat)
    return;

  const float4 val = from[idat];
  to[idat] = make_float2(val.x, val.y);
  to[idat + out_pol_stride] = make_float2(val.z, val.y);
}

//! unpacks npol=1 inputs from float2 (Re,Im)
__global__ void uwb_unpack_t_fpt_float_kernel(
  float2 * to,
  const float2 * from,
  uint64_t ndat
)
{
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idat >= ndat)
    return;

  to[idat] = from[idat];
}

__global__ void uwb_unpack_pt_fpt_float_kernel (float2 * to,
                                             const float2 * from,
                                             uint64_t out_pol_stride,
                                             unsigned nblock,
                                             uint64_t ndat_per_block)
{
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idat >= ndat_per_block)
    return;

  const unsigned ipol = blockIdx.z;
  const unsigned npol = gridDim.z;

  // SFPT ordering, S==1, F==1, so PT
  uint64_t idx = (ipol * ndat_per_block) + idat;
  uint64_t odx = (ipol * out_pol_stride) + idat;

  // block_stride is               npol * ndat_per_block
  const uint64_t in_block_stride = npol * ndat_per_block;

  for (unsigned i=0; i<nblock; i++)
  {
    to[odx] = from[idx];

    idx += in_block_stride;
    odx += ndat_per_block;
  }
}

CUDA::UWBFloatUnpackerEngine::UWBFloatUnpackerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::UWBFloatUnpackerEngine::setup ()
{
  // determine cuda device properties for block & grid size
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

bool CUDA::UWBFloatUnpackerEngine::get_device_supported (dsp::Memory* memory) const
{
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
}

void CUDA::UWBFloatUnpackerEngine::set_device (dsp::Memory* memory)
{
}

void CUDA::UWBFloatUnpackerEngine::unpack (const dsp::BitSeries * input, dsp::TimeSeries * output, const std::string& input_order)
{
  const uint64_t ndat = input->get_ndat();
  const int npol = input->get_npol();
  const int nchan = input->get_nchan();

  if (nchan > 1)
    throw Error(InvalidState, "CUDA::UWBFloatUnpackerEngine::unpack", "Only nchan == 1 supported");

#ifdef _DEBUG
  cerr << "CUDA::UWBFloatUnpackerEngine::unpack ndat=" << ndat << " npol=" << npol << " nchan=" << nchan << endl;
#endif

  // use an float2 to handle the re+im parts of the float
  float2 * from = (float2 *) input->get_rawptr();
  float2 * into = (float2 *) output->get_datptr(0, 0);
  uint64_t pol_stride = 0;

  if (npol == 2)
  {
    float2 * into_a = (float2 *) output->get_datptr(0, 0);
    float2 * into_b = (float2 *) output->get_datptr(0, 1);
    pol_stride = uint64_t(into_b - into_a);
  }

  if (input_order == "TFPS")
  {
    unsigned nthreads = 1024;
    dim3 blocks = dim3 (ndat / nthreads, 1, 1);
    if (ndat % nthreads > 0)
      blocks.x++;

    if (npol == 2)
    {
      uwb_unpack_tp_fpt_float_kernel<<<blocks,nthreads,0,stream>>> (into, reinterpret_cast<float4*>(from), ndat, pol_stride);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream ("CUDA::UWBFloatUnpackerEngine::unpack uwb_unpack_tp_fpt_float_kernel", stream);
    }
    else
    {
      uwb_unpack_t_fpt_float_kernel<<<blocks,nthreads,0,stream>>> (into, reinterpret_cast<float2*>(from), ndat);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream ("CUDA::UWBFloatUnpackerEngine::unpack uwb_unpack_t_fpt_float_kernel", stream);
    }
  }
  else
  {
    const unsigned ndat_per_block = input->get_loader()->get_resolution();
    const uint64_t nblock = ndat / ndat_per_block;

    if (ndat % ndat_per_block)
      throw Error(InvalidState, "CUDA::UWBFloatUnpackerEngine::unpack", "ndat was not divisible by resolution");

    unsigned nthreads = 1024;
    dim3 blocks = dim3 (ndat_per_block / nthreads, nchan, npol);
    if (ndat_per_block % nthreads > 0)
      blocks.x++;

#ifdef _DEBUG
    cerr << "CUDA::UWBFloatUnpackerEngine::unpack nthreads=" << nthreads
         << " blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z
         << ")" << endl;
#endif

    uwb_unpack_pt_fpt_float_kernel<<<blocks,nthreads,0,stream>>> (into, from, pol_stride, nblock, ndat_per_block);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream ("CUDA::UWBFloatUnpackerEngine::unpack uwb_unpack_pt_fpt_float_kernel", stream);

  }
}
