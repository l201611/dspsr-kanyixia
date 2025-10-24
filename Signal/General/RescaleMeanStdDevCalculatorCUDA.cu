//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/RescaleMeanStdDevCalculatorCUDA.h"

#include <assert.h>
#include <cstring>

#define FULLMASK 0xFFFFFFFF

using namespace std;

static constexpr unsigned nthreads = 1024;
static constexpr unsigned warp_size = 32;

void check_error_stream(const char *, cudaStream_t);

template<typename T>
void zero (vector<T>& data)
{
  std::fill(data.begin(), data.end(), 0);
}

CUDA::RescaleMeanStdDevCalculatorCUDA::RescaleMeanStdDevCalculatorCUDA(cudaStream_t _stream) : CUDA::ScaleOffsetCalculatorCUDA(_stream)
{
  d_freq_total = h_freq_total = nullptr;
  d_freq_totalsq = h_freq_totalsq = nullptr;
}

CUDA::RescaleMeanStdDevCalculatorCUDA::~RescaleMeanStdDevCalculatorCUDA()
{
  CUDA::RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_freq_total));
  CUDA::RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_freq_totalsq));

  CUDA::RescaleEngine::release_host_memory(reinterpret_cast<void **>(&h_freq_total));
  CUDA::RescaleEngine::release_host_memory(reinterpret_cast<void **>(&h_freq_totalsq));
}

/*
 Compute a sum of a float across a warp.

 This is a utility kernel to reduce the sum of a value across a warp.

 @param val the value for a given thread within a warp
 @returns the value summed across all threads in the warp.
 */
__inline__ __device__ float rescale_mean_std_warp_reduce_sum(float val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
  {
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
    val += __shfl_down_sync(FULLMASK, val, offset);
#else
    val += __shfl_down(val, offset);
#endif
#endif
  }
  return val;
}

/*
  Compute a sum of a value across a whole block.

  This is utility kernel to help getting a sum of a value across a block.

  @param val the value for a given thread within a block
  @returns the value summed across all threads in the block.
 */
__inline__ __device__ float rescale_mean_std_block_reduce_sum(float val)
{
  // shared mem for 32 partial sums
  __shared__ float shared[32];

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  // each warp performs partial reduction
  val = rescale_mean_std_warp_reduce_sum(val);

  // write reduced value to shared memory
  if (lane == 0)
    shared[wid] = val;

  // wait for all partial reductions
  __syncthreads();

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  // final reduce within first warp
  if (wid == 0)
    val = rescale_mean_std_warp_reduce_sum(val);

  return val;
}

/**
 * @brief sample the data needed to calculate the freq total and freq squared totals for TFP-ordered data.
 *
 * @param in_ptr base address of TimeSeries from where data statitics are to be calculated from
 * @param freq_total the base address of where to output the frequency totals are stored in FP ordering.
 * @param freq_totalsq the base address of where the output of the frequency squared totals are stored in FP order.
 * @param ndat the total number of time samples to be used to calculate the statistics.
 * @param nchan the number of channels that the input timeseries has.
 * @param npol the number of polarizations that the input timeseries has.
 * @param ndim the number of dimensions the samples in the input timeseries has.
 *
 * number of CUDA blocks - typically ceil(nchan * npol / warpSize)
 * each warp processes exactly 1 channel+pol (ichanpol) combination and each block of 1024 threads will process
 * warpSize ichanpols.
 *
 * warp_idx = threadIdx.x % warpSize - the index of the current thread within a warp
 * warp_num = threadIdx.x / warpSize - a number representing which warp within block the thread belongs to
 *
 * ichanpol = (blockIdx.x * warpSize) + warp_num - which channel + pol combination the thread is working on.
 */
__global__ void rescale_mean_std_sample_data_tfp(const float *in_ptr, float *freq_total, float *freq_totalsq, unsigned ndat, unsigned nchan, unsigned npol, unsigned ndim)
{
  const unsigned nchanpol = nchan * npol;
  const unsigned warp_idx = threadIdx.x % warpSize;
  const unsigned warp_num = threadIdx.x / warpSize;

  // each warp processes 1 channel, each block of 1024 processes 32 channels
  unsigned ichanpol = (blockIdx.x * warpSize) + warp_num;

  // the sample offset for this thread
  uint64_t idx = ((warp_idx * nchanpol) + ichanpol) * ndim;

  float freq_total_thread = 0.0;
  float freq_totalsq_thread = 0.0;

  if (ichanpol < nchanpol)
  {
    const uint64_t warp_stride = nchanpol * ndim * warpSize;

    // process all of the samples for this chan/pol
    for (uint64_t idat = warp_idx; idat < ndat; idat += warpSize)
    {
      for (unsigned idim = 0; idim < ndim; idim++)
      {
        const float in_val = in_ptr[idx + idim];
        freq_total_thread += in_val;
        freq_totalsq_thread += (in_val * in_val);
      }
      idx += warp_stride;
    }

    // now reduce across the warp
    freq_total_thread = rescale_mean_std_warp_reduce_sum(freq_total_thread);

    freq_totalsq_thread = rescale_mean_std_warp_reduce_sum(freq_totalsq_thread);
    __syncthreads();

    // store freq_total, freq_totalsq, offset and scale in FP order
    if (warp_idx == 0)
    {
      freq_total[ichanpol] += freq_total_thread;
      freq_totalsq[ichanpol] += freq_totalsq_thread;
    }
  }
}

/**
 * @brief sample the data needed to calculate the freq total and freq squared totals for FPT-ordered data.
 *
 * @param in_ptr base address of TimeSeries from where data statistics are to be calculated from
 * @param chanpol_stride the stride in data between a channel and a polarisation.
 * @param freq_total the base address of where to output the frequency totals are stored in FP ordering.
 * @param freq_totalsq the base address of where the output of the frequency squared totals are stored in FP order.
 * @param start_dat the starting time sample to get sample from, usually 0 but may not be.
 * @param ndat the total number of time samples to be used to calculate the statistics.
 * @param ndim the number of dimensions the samples in the input timeseries has.
 *
 * number of CUDA blocks - typically nchan * npol
 * ichanpol = blockIdx.x
 *
 * each warp processes exactly 1 channel+pol (ichanpol) combination and each block of 1024 threads will process
 * warpSize ichanpols.
 */
__global__ void rescale_mean_std_sample_data_fpt(const float *in, unsigned chanpol_stride,
                                              float *freq_total, float *freq_totalsq,
                                              uint64_t start_dat, uint64_t ndat, unsigned ndim)
{
  const unsigned ichanpol = blockIdx.x;
  const uint64_t chanpol_offset = ichanpol * chanpol_stride;

  float freq_total_thread = 0;
  float freq_totalsq_thread = 0;

  const float *in_ptr = in + chanpol_offset;

  for (uint64_t idat = threadIdx.x; idat < ndat; idat += blockDim.x)
  {
    uint64_t idat_idx = (idat + start_dat) * ndim;
    for (unsigned idim = 0; idim < ndim; idim++)
    {
      const float in_val = in_ptr[idat_idx + idim];
      freq_total_thread += in_val;
      freq_totalsq_thread += (in_val * in_val);
    }
  }

  // sum across block
  freq_total_thread = rescale_mean_std_block_reduce_sum(freq_total_thread);

  // force a sync here so since shared memory is shared in the 2 reductions
  __syncthreads();

  // sum across block
  freq_totalsq_thread = rescale_mean_std_block_reduce_sum(freq_totalsq_thread);

  __syncthreads();

  if (threadIdx.x == 0)
  {
    freq_total[ichanpol] += freq_total_thread;
    freq_totalsq[ichanpol] += freq_totalsq_thread;
  }
}

/**
 * @brief the offsets, scales from sampled data.
 *
 * The freq_total and freq_totalsq are in FP ordering and this kernel can be used for both FPT and TFP
 * input data as the sampling kernels have already provided the logic to get the totals.
 *
 * @param freq_total the base address of where to get the frequency totals are stored in FP ordering.
 * @param freq_totalsq the base address of where the get of the frequency squared totals are stored in FP order.
 * @param offset the base address of where to output the offsets in FP ordering.
 * @param scale the base address of where to output the scales in FP ordering.
 * @param ndat the total number of time samples to be used to calculate the statistics.
 * @param nchan the number of channels that the input timeseries has.
 * @param npol the number of polarizations that the input timeseries has.
 * @param ndim the number of dimensions the samples in the input timeseries has.
 *
 * The scale is 1.0/sqrt(variance) while the offset is the negative value of the mean.  This is consistent with
 * the CPU implementation.
 *
 * recip is typically 1.0 / (ndat * ndim)
 *
 * number of CUDA blocks - typically ceil(nchan * npol / warpSize)
 * each warp processes exactly 1 channel+pol (ichanpol) combination and each block of 1024 threads will process
 * warpSize ichanpols.
 *
 * warp_idx = threadIdx.x % warpSize - the index of the current thread within a warp
 * warp_num = threadIdx.x / warpSize - a number representing which warp within block the thread belongs to
 *
 * ichanpol = (blockIdx.x * warpSize) + warp_num - which channel + pol combination the thread is working on.
 */
__global__ void rescale_mean_std_calculate(const float *freq_total, const float *freq_totalsq, float *offset, float *scale, unsigned ndat, unsigned nchan, unsigned npol, unsigned ndim, float recip)
{
  const unsigned nchanpol = nchan * npol;
  const unsigned warp_num = threadIdx.x / warpSize;

  // each warp processes 1 channel, each block of 1024 processes 32 channels
  unsigned ichanpol = (blockIdx.x * warpSize) + warp_num;

  if (ichanpol < nchanpol)
  {
    const float mean = freq_total[ichanpol] * recip;
    const float variance = freq_totalsq[ichanpol] * recip - mean * mean;

    offset[ichanpol] = -mean;
    if (variance <= 0.0)
      scale[ichanpol] = 1.0;
    else
      scale[ichanpol] = 1.0 / sqrt(variance);
  }
}

void CUDA::RescaleMeanStdDevCalculatorCUDA::init(const dsp::TimeSeries* input, uint64_t _ndat, bool output_time_total)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::init started" << endl;

  gpu_config.init();

  ndat = _ndat;
  npol = input->get_npol();
  ndim = input->get_ndim();
  nchan = input->get_nchan();

  data_size_bytes = npol * nchan * sizeof(float);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::init calling init_common_memory" << endl;

  init_common_memory(data_size_bytes);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::init allocating device ptrs " << endl;

  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_freq_total), data_size_bytes);
  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_freq_totalsq), data_size_bytes);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::init initialising device ptrs. data_size_bytes=" << data_size_bytes << endl;

  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_freq_total), data_size_bytes, stream);
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::RescaleMeanStdDevCalculatorCUDA::init - initialising d_freq_total", stream);

  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_freq_totalsq), data_size_bytes, stream);
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::RescaleMeanStdDevCalculatorCUDA::init - initialising d_freq_totalsq", stream);

  // allocate host memory
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::init allocating host ptrs" << endl;

  RescaleEngine::allocate_host_memory(reinterpret_cast<void **>(&h_freq_total), data_size_bytes);
  RescaleEngine::allocate_host_memory(reinterpret_cast<void **>(&h_freq_totalsq), data_size_bytes);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::init initialing host ptrs" << endl;

  RescaleEngine::init_host_memory(reinterpret_cast<void *>(h_freq_total), data_size_bytes);
  RescaleEngine::init_host_memory(reinterpret_cast<void *>(h_freq_totalsq), data_size_bytes);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::init setting up host data vectors" << endl;

  freq_total.resize (npol);
  freq_totalsq.resize (npol);
  scale.resize (npol);
  offset.resize (npol);

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    freq_total[ipol].resize (nchan, 0);
    freq_totalsq[ipol].resize (nchan, 0);
    scale[ipol].resize (nchan, 0);
    offset[ipol].resize (nchan, 0);
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::init finished" << endl;

}

void CUDA::RescaleMeanStdDevCalculatorCUDA::reset_sample_data()
{
  ScaleOffsetCalculatorCUDA::reset_sample_data();

  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_freq_total), data_size_bytes, stream);
  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_freq_totalsq), data_size_bytes, stream);
  cudaStreamSynchronize(stream);

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    zero (freq_total[ipol]);
    zero (freq_totalsq[ipol]);
  }

  nintegrated = 0;
}

auto CUDA::RescaleMeanStdDevCalculatorCUDA::sample_data(const dsp::TimeSeries* input, uint64_t start_dat, uint64_t end_dat, bool output_time_total) -> uint64_t
{
  auto nchanpol = nchan * npol;
  auto nsamp = end_dat - start_dat;

  if (nsamp == 0)
    return nintegrated;

  switch (input->get_order())
  {
  case dsp::TimeSeries::OrderTFP:
  {
    const float *in_ptr = input->get_dattfp();
    uint64_t ptr_offset = start_dat * nchanpol * ndim;

    auto nblocks = nchanpol / warp_size;
    if (nchanpol % warp_size != 0)
    {
      nblocks++;
    }

    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::sample_data calling rescale_mean_std_sample_data_tfp nblocks=" << nblocks
            << ", nthreads=" << nthreads << ", ptr_offset=" << ptr_offset << endl;

    rescale_mean_std_sample_data_tfp<<<nblocks, nthreads, 0, stream>>>(in_ptr + ptr_offset, d_freq_total, d_freq_totalsq, nsamp, nchan, npol, ndim);

    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream("CUDA::RescaleMeanStdDevCalculatorCUDA::sample_data rescale_mean_std_sample_data_tfp", stream);

    break;
  }
  case dsp::TimeSeries::OrderFPT:
  {
    auto nblocks = nchanpol;
    auto chanpol_stride = input->get_stride();

    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::sample_data calling rescale_mean_std_sample_data_fpt nblocks=" << nblocks
            << ", nchanpol=" << nchanpol << ", nchan=" << nchan << ", npol=" << npol
            << ", nthreads=" << nthreads << ", chanpol_stride=" << chanpol_stride << endl;

    rescale_mean_std_sample_data_fpt<<<nblocks, nthreads, 0, stream>>>(input->get_datptr(0, 0), chanpol_stride,
                                                                    d_freq_total, d_freq_totalsq,
                                                                    start_dat, nsamp, ndim);

    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream("CUDA::RescaleMeanStdDevCalculatorCUDA::sample_data rescale_mean_std_sample_data_fpt", stream);

    break;
  }
  }

  nintegrated += nsamp;
  return nintegrated;
}

void CUDA::RescaleMeanStdDevCalculatorCUDA::compute()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::compute started" << endl;

  float recip{0.0};
  if (nintegrated > 0)
  {
    recip = 1.0 / static_cast<float>(nintegrated * ndim);
  }

  auto nchanpol = nchan * npol;
  auto nblocks = nchanpol / warp_size;
  if (nchanpol % warp_size != 0)
  {
    nblocks++;
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::compute calling rescale_mean_std_calculate nblocks=" << nblocks
          << ", nthreads=" << nthreads << ", recip=" << recip << endl;

  rescale_mean_std_calculate<<<nblocks, nthreads, 0, stream>>>(d_freq_total, d_freq_totalsq, d_offset, d_scale, nintegrated, nchan, npol, ndim, recip);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::RescaleMeanStdDevCalculatorCUDA::compute rescale_mean_std_calculate", stream);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMeanStdDevCalculatorCUDA::compute exiting" << endl;
}

void CUDA::RescaleMeanStdDevCalculatorCUDA::transfer_to_host()
{
  cudaError_t error;

  error = cudaMemcpyAsync(h_offset, d_offset, data_size_bytes, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleMeanStdDevCalculatorCUDA::compute", "cudaMemcpyAsync from d_offset to h_offset failed");

  error = cudaMemcpyAsync(h_scale, d_scale, data_size_bytes, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleMeanStdDevCalculatorCUDA::compute", "cudaMemcpyAsync from d_scale to h_scale failed");

  error = cudaMemcpyAsync(h_freq_total, d_freq_total, data_size_bytes, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleMeanStdDevCalculatorCUDA::compute", "cudaMemcpyAsync from d_freq_total to h_freq_total failed");

  error = cudaMemcpyAsync(h_freq_totalsq, d_freq_totalsq, data_size_bytes, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleMeanStdDevCalculatorCUDA::compute", "cudaMemcpyAsync from d_freq_totalsq to h_freq_totalsq failed");

  // this will perform a cudaStreamSynchronize and handle the error
  check_error_stream("CUDA::RescaleMeanStdDevCalculatorCUDA::compute stream sync", stream);

  // Need to transpose from FP to PF and store in std::vectors
  for (auto ipol = 0; ipol < npol; ipol++)
  {
    for (auto ichan = 0; ichan < nchan; ichan++)
    {
      auto ichanpol = ichan * npol + ipol;
      offset[ipol][ichan] = h_offset[ichanpol];
      scale[ipol][ichan] = h_scale[ichanpol];
      freq_total[ipol][ichan] = static_cast<double>(h_freq_total[ichanpol]);
      freq_totalsq[ipol][ichan] = static_cast<double>(h_freq_totalsq[ichanpol]);
    }
  }
}
