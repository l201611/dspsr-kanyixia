//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015-2025 by Matthew Kerr and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TScrunchCUDA.h"

#include <cuComplex.h>
#include "Error.h"
#include "debug.h"

using namespace std;

void check_error_stream (const char*, cudaStream_t stream);

CUDA::TScrunchEngine::TScrunchEngine (cudaStream_t _stream)
{
  stream = _stream;
  gpu_config.init();
}

#if (__CUDA_ARCH__ >= 300)
// to support templated kernels below
__inline__ __device__ float warpSum(float val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
#if (__CUDACC_VER_MAJOR__>= 9)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
#else
    val += __shfl_down(val, offset);
#endif
  return val;
}

__inline__ __device__ float2 warpSum(float2 val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
  {
#if (__CUDACC_VER_MAJOR__ >= 9)
    val.x += __shfl_down_sync(0xFFFFFFFF, val.x, offset);
    val.y += __shfl_down_sync(0xFFFFFFFF, val.y, offset);
#else
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
#endif
  }
  return val;
}
#endif

__inline__ __device__ float sumTwo(float v1, float v2)
{
  return v1 + v2;
}

__inline__ __device__ float2 sumTwo(float2 v1, float2 v2)
{
  return cuCaddf(v1, v2);
}

__inline__ __device__ void initVal(float * v)
{
  *v = 0;
}

__inline__ __device__ void initVal(float2 * v)
{
  (*v).x = 0;
  (*v).y = 0;
}

// each warp writes 1 output sample via load + shuffle
template<typename T>
__global__ void fpt_warp (T* in_base, T* out_base,
    unsigned in_Fstride, unsigned in_Pstride,
    unsigned out_Fstride, unsigned out_Pstride,
    unsigned ndat_out, unsigned odat_per_block, unsigned sfactor)
{
  __shared__ T ndim1_warp_shm[16];

  const unsigned warp_num = threadIdx.x / warpSize;
  const unsigned warp_idx = threadIdx.x % warpSize;
  const unsigned odat_block_offset = blockIdx.x * odat_per_block;

  uint64_t odat = odat_block_offset + warp_num;
  T sum;
  initVal(&sum);

  if (odat < ndat_out)
  {
    // offset into buffer = the index the first read sample for this block
    in_base += (blockIdx.y*in_Fstride) + (blockIdx.z*in_Pstride) + (odat * sfactor);

    unsigned isamp = warp_idx;

    // first sum in each thread of the warp independently (good if sfactor > 32)
    while (isamp < sfactor)
    {
      sum = sumTwo(sum, in_base[isamp]);
      //sum = sum + in_base[isamp];
      isamp += warpSize;
    }

    // sum across the warp
#if (__CUDA_ARCH__ >= 300)
    sum = warpSum(sum);
#endif
  }

  // only the first thread in each warp sum writes out
  if (warp_idx == 0)
  {
    ndim1_warp_shm[warp_num] = sum;
  }

  __syncthreads();

  // only the first warp writes out the sums
  if (warp_num == 0 && warp_idx < 16)
  {
    uint64_t odat = odat_block_offset + warp_idx;
    if (odat < ndat_out)
    {
      out_base += blockIdx.y * out_Fstride + blockIdx.z * out_Pstride;
      out_base[odat] = ndim1_warp_shm[warp_idx];
    }
  }
}

// each thread processes 1 output sample with all block samples loaded into SHM
template<typename T>
__global__ void fpt_scrunch_shared (T* in_base, T* out_base,
    unsigned in_Fstride, unsigned in_Pstride,
    unsigned out_Fstride, unsigned out_Pstride,
    unsigned ndat_out, unsigned sfactor, unsigned nchan)
{
  // 256 threads x 16 samples
  __shared__ T ndim1_shared_shm[4096];

  const uint64_t odat_block = blockIdx.x * blockDim.x;

  for (unsigned ichan=blockIdx.y; ichan<nchan; ichan+=gridDim.y)
  {
    // offset into buffer = the index the first read sample for this block
    uint64_t idx = (ichan*in_Fstride) + (blockIdx.z*in_Pstride) + (odat_block * sfactor);

    // load data for all threads in coalesced manner
    uint64_t idat = (odat_block * sfactor) + threadIdx.x;
    unsigned sdx = threadIdx.x;
    const uint64_t ndat = ndat_out * sfactor;
    for (unsigned i=0; i<sfactor; i++)
    {
      if (idat < ndat)
        ndim1_shared_shm[sdx] = in_base[idx + sdx];
      else
        initVal(ndim1_shared_shm + sdx);
      sdx += blockDim.x;
      idat += blockDim.x;
    }

    __syncthreads();

    // output dat
    const uint64_t odat = odat_block + threadIdx.x;
    if (odat < ndat_out)
    {
      uint64_t odx = (ichan*out_Fstride) + (blockIdx.z*out_Pstride);

      // now each thread reads shm, bank conflicts are unavoidable
      T sum;
      initVal(&sum);

      unsigned sdx_offset = threadIdx.x * sfactor;
      for (unsigned i=0; i<sfactor; i++)
        sum = sumTwo(sum, ndim1_shared_shm[sdx_offset + i]);

      // each thread writes the output
      out_base[odx + odat] = sum;
    }
  }
}

void CUDA::TScrunchEngine::fpt_tscrunch(const dsp::TimeSeries *in,
    dsp::TimeSeries* out, unsigned sfactor)
{
  // split the the processing between 2 algorithms. If the tscrunch factor
  // is > 8, then use warps to perform the required scrunching. If < 8 use
  // shared memory with 1 thread per output sample
  if (in->get_ndim() != out->get_ndim())
  {
    throw Error (InvalidParam, "CUDA::TScrunchEngine::fpt_scrunch",
       "cannot handle input ndim=%u != output ndim=%u", in->get_ndim()),
       out->get_ndim();
  }

  if (out == in)
    throw Error (InvalidParam, "CUDA::TScrunchEngine::fpt_scrunch",
		 "only out-of-place transformation implemented");

  if (in->get_ndat() == 0 || out->get_ndat() == 0)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::TScrunchEngine::fpt_scrunch in_ndat=" << in->get_ndat()
           << " out_ndat=" << out->get_ndat() << ", skipping" << endl;
    return;
  }

  unsigned ndim = in->get_ndim();

  uint64_t in_Fstride = 0;
  uint64_t out_Fstride = 0;
  if (in->get_nchan() > 1)
  {
    in_Fstride = (in->get_datptr(1,0)-in->get_datptr(0,0)) / ndim;
    out_Fstride = (out->get_datptr(1,0)-out->get_datptr(0,0)) / ndim;
  }

  uint64_t in_Pstride = 0;
  uint64_t out_Pstride = 0;
  if (in->get_npol() > 1)
  {
    in_Pstride = (in->get_datptr(0,1)-in->get_datptr(0,0)) / ndim;
    out_Pstride = (out->get_datptr(0,1)-out->get_datptr(0,0)) / ndim;
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::TScrunchEngine::fpt_scrunch ndim=" << ndim
         << " in_Fstride=" << in_Fstride << " out_Fstride=" << out_Fstride
         << " in_Pstride=" << in_Pstride << " out_Pstride=" << out_Pstride
         << endl;

  if (sfactor >= 16)
  {
    unsigned nthreads = 512;
    unsigned odat_per_block = nthreads / 32;
    dim3 blocks (out->get_ndat()/odat_per_block, in->get_nchan(), in->get_npol());
    if (out->get_ndat() % odat_per_block)
      blocks.x ++;

    if (dsp::Operation::verbose)
      cerr << "CUDA::TScrunchEngine::fpt_warp blocks=(" << blocks.x << ","
         << blocks.y << "," << blocks.z << ") nthreads=" << nthreads << endl;

    if (ndim == 1)
      fpt_warp<float><<<blocks,nthreads,0,stream>>> (
        (float*)(in->get_datptr(0)), (float*)(out->get_datptr(0)),
        in_Fstride, in_Pstride, out_Fstride, out_Pstride,
        out->get_ndat(), odat_per_block, sfactor
      );
    else
      fpt_warp<float2><<<blocks,nthreads,0,stream>>> (
        (float2*)(in->get_datptr(0)), (float2*)(out->get_datptr(0)),
        in_Fstride, in_Pstride, out_Fstride, out_Pstride,
        out->get_ndat(), odat_per_block, sfactor
      );
    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream ("CUDA::TScrunchEngine::fpt_scrunch_warp", stream);
  }
  else
  {
    unsigned nthreads = 256;
    dim3 blocks (out->get_ndat()/nthreads, in->get_nchan(), in->get_npol());
    if (out->get_ndat() % nthreads)
      blocks.x ++;
    if (blocks.y > gpu_config.get_max_grid_size_y())
      blocks.y = gpu_config.get_max_grid_size_y();

    if (dsp::Operation::verbose)
      cerr << "CUDA::TScrunchEngine::fpt_scrunch_shared blocks=(" << blocks.x << ","
         << blocks.y << "," << blocks.z << ") nthreads=" << nthreads << endl;

    if (ndim == 1)
      fpt_scrunch_shared<float><<<blocks,nthreads,0,stream>>> (
        (float*)(in->get_datptr(0)), (float*)(out->get_datptr(0)),
        in_Fstride, in_Pstride, out_Fstride, out_Pstride,
        out->get_ndat(), sfactor, in->get_nchan());
    else
      fpt_scrunch_shared<float2><<<blocks,nthreads,0,stream>>> (
        (float2*)(in->get_datptr(0)), (float2*)(out->get_datptr(0)),
        in_Fstride, in_Pstride, out_Fstride, out_Pstride,
        out->get_ndat(), sfactor, in->get_nchan());
  }
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::TScrunchEngine::fpt_scrunch_shared", stream);
}

__global__ void tfp_scrunch (
  const float * __restrict__ input, float * output,
  unsigned nchanpoldim, unsigned sfactor)
{
  const unsigned ichanpoldim = (blockIdx.y * blockDim.x) + threadIdx.x;
  if (ichanpoldim >= nchanpoldim)
    return;

  // odat == blockIdx.x
  const unsigned idat = blockIdx.x * sfactor;

  // offset into buffer = the index the first read sample for this block
  uint64_t idx = (idat * nchanpoldim) + ichanpoldim;

  float sum = 0;
  for (unsigned i=0; i<sfactor; i++)
  {
    sum += input[idx];
    idx += nchanpoldim;
  }

  output[blockIdx.x * nchanpoldim + ichanpoldim] = sum;
}

__global__ void tfp_scrunch_small_chanpoldim (
  const float * __restrict__ input, float * output,
  uint64_t output_ndat, unsigned sfactor)
{
  // blockDim.x == nchanpoldim
  const unsigned nchanpoldim = blockDim.x;
  const unsigned ichanpoldim = threadIdx.x;

  // odat == blockIdx.x * blockDim.y + threadIdx.y
  const unsigned odat = blockIdx.x * blockDim.y + threadIdx.y;
  const unsigned idat = odat * sfactor;

  // printf("[%d][%d] ichanpoldim=%u odat=%u idat=%u\n", threadIdx.x, threadIdx.y, ichanpoldim, odat, idat);

  // offset into buffer = the index the first read sample for this block
  uint64_t idx = (idat * nchanpoldim) + ichanpoldim;

  float sum = 0;
  for (unsigned i=0; i<sfactor; i++)
  {
    sum += input[idx];
    idx += nchanpoldim;
  }

  output[odat * nchanpoldim + ichanpoldim] = sum;
}

void CUDA::TScrunchEngine::tfp_tscrunch(const dsp::TimeSeries *in, dsp::TimeSeries* out, unsigned sfactor)
{
  if (in->get_ndim() != out->get_ndim())
  {
    throw Error (InvalidParam, "CUDA::TScrunchEngine::tfp_tscrunch",
      "cannot handle input ndim=%u != output ndim=%u", in->get_ndim()),
      out->get_ndim();
  }

  if (out == in)
    throw Error (InvalidParam, "CUDA::TScrunchEngine::tfp_tscrunch",
    "only out-of-place transformation supported");

  if (in->get_ndat() == 0 || out->get_ndat() == 0)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::TScrunchEngine::tfp_tscrunch in_ndat=" << in->get_ndat()
          << " out_ndat=" << out->get_ndat() << ", skipping" << endl;
    return;
  }

  // assuming a minimum viable block size of 128 threads
  unsigned nchanpoldim = in->get_nchan() * in->get_npol() * in->get_ndim();
  uint64_t ndat = in->get_ndat();

  auto input = reinterpret_cast<const float *>(in->get_dattfp());
  auto output = reinterpret_cast<float *>(out->get_dattfp());

  if (dsp::Operation::verbose)
    cerr << "CUDA::TScrunchEngine::tfp_tscrunch ndat=" << ndat << " nchanpoldim=" << nchanpoldim << endl;

  // the smallest number of channels * polarisations * dimensions that is efficient for the tfp_scrunch kernel
  static constexpr unsigned min_nchanpoldim = 32;

  if (nchanpoldim >= min_nchanpoldim)
  {
    unsigned nthreads = std::min(static_cast<unsigned>(512), nchanpoldim);
    uint64_t output_dat = ndat / sfactor;
    dim3 blocks = dim3(output_dat, nchanpoldim / nthreads, 1);
    if (nchanpoldim % nthreads != 0)
    {
      blocks.y++;
    }

    if (dsp::Operation::verbose)
      cerr << "CUDA::TScrunchEngine::tfp_tscrunch tfp_scrunch kernel blocks=(" <<  blocks.x << "," << blocks.y << "," << blocks.z << ") nthreads=" << nthreads << endl;
    tfp_scrunch<<<blocks, nthreads, 0, stream>>> (input, output, nchanpoldim, sfactor);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
    {
      check_error_stream ("CUDA::TScrunchEngine::tfp_scrunch", stream);
    }
  }
  else
  {
    // threads.x each handle a chanpoldim
    // threads.y each handle a different time sample
    unsigned output_ndat = ndat / sfactor;
    unsigned output_ndat_per_block = std::min(output_ndat, 512 / nchanpoldim);
    dim3 threads = dim3(nchanpoldim, output_ndat_per_block, 1);
    unsigned nblocks = output_ndat / threads.y;
    if (output_ndat % threads.y != 0)
    {
      nblocks++;
    }
    if (dsp::Operation::verbose)
      cerr << "CUDA::TScrunchEngine::tfp_tscrunch tfp_scrunch_small_chanpoldim kernel nblocks=" << nblocks << " threads(" <<  threads.x << "," << threads.y << "," << threads.z << ")" << endl;
    tfp_scrunch_small_chanpoldim<<<nblocks, threads, 0, stream>>> (input, output, output_ndat, sfactor);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
    {
      check_error_stream ("CUDA::TScrunchEngine::tfp_scrunch_small_chanpoldim", stream);
    }
  }
}
