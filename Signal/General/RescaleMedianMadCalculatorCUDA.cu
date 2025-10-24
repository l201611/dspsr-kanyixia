//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/RescaleMedianMadCalculatorCUDA.h"

#include <assert.h>
#include <cstring>
#include <cub/device/device_segmented_sort.cuh>

#define FULLMASK 0xFFFFFFFF

using namespace std;

static constexpr unsigned nthreads = 1024;

void check_error_stream(const char *, cudaStream_t);

template<typename T>
void zero (vector<T>& data)
{
  std::fill(data.begin(), data.end(), 0);
}

CUDA::RescaleMedianMadCalculatorCUDA::RescaleMedianMadCalculatorCUDA(cudaStream_t _stream) : CUDA::ScaleOffsetCalculatorCUDA(_stream)
{
}

CUDA::RescaleMedianMadCalculatorCUDA::~RescaleMedianMadCalculatorCUDA()
{
  CUDA::RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_sample_data));
  CUDA::RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_absolute_deviation));
  CUDA::RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_sorted));
  CUDA::RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_temp_storage));
  CUDA::RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_chanpol_start_offset));
  CUDA::RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_chanpol_end_offset));

  CUDA::RescaleEngine::release_host_memory(reinterpret_cast<void **>(&h_chanpol_start_offset));
  CUDA::RescaleEngine::release_host_memory(reinterpret_cast<void **>(&h_chanpol_end_offset));
}

/**
 * @brief CUDA kernel to calculate the offset of the sorted input data.
 *
 * The offset of the data is the negative median (i.e. -median) of the
 * sorted input data. This kernel calculates the median of each channel/polarisation
 * combination from sorted data and stores the offset
 *
 * This doesn't find the correct median when NDAT is even, because technically it
 * would have to be the average of 2 values but this takes the value with index of (ndat - 1)/2,
 * using 0-offset indexing.
 *
 * This kernel also calculates the absolute deviation of sorted data from the median
 * which can than be used to find the MAD by later sorting the data.
 *
 * @param sorted the sorted data grouped by channel and polarisation
 * @param offset the output array to store the computed offset (i.e. -median) for
 *  each channel and polarisation
 * @param absolute_deviation the output array to store the calculated absolute deviation
 *  of a channel and polarisation
 * @param ndat the number of time samples in each channel/polarisation to get the median value from
 * @param chanpol_stride the number of floating point values per channel/polarisation, this
 *  total_nsamp * ndim
 * @param median_idx the index to the median value of a sorted chan/pol data segment, this is (nintegrated * ndim - 1)/2
 */
__global__ void rescale_calculate_offset(const float * sorted, float *offset, float *absolute_deviation, const uint64_t ndat, const uint64_t chanpol_stride, const uint64_t median_idx)
{
  const unsigned ichanpol = blockIdx.x;
  const uint64_t chanpol_offset = ichanpol * chanpol_stride;

  sorted += chanpol_offset;
  absolute_deviation += chanpol_offset;

  const float median = sorted[median_idx];
  if (threadIdx.x == 0)
  {
    offset[ichanpol] = -median;
  }

  for (uint64_t idat = threadIdx.x; idat < ndat; idat += blockDim.x)
  {
    absolute_deviation[idat] = fabsf(sorted[idat] - median);
  }
}

/**
 * @brief CUDA kernel to calculate the scale of the data based on the median absolute deviation (MAD).
 *
 * @param sorted the sorted data grouped by channel and polarisation, this should be sorted data
 *  of the absolute deviation from the median.
 * @param scale the output array to store the computed scale for each channel and polarisation
 * @param nchanpol the total number of channel/polarisation combinations
 * @param chanpol_stride the number of floating point values per channel/polarisation, this
 *  total_nsamp * ndim
 * @param scale_factor the scale factor to apply to convert MAD to an estimate of the standard deviation.
 * @param median_idx the index to the median value of a sorted chan/pol data segment, this is (nintegrated * ndim - 1)/2
 */
__global__ void rescale_calculate_scale(const float *sorted, float *scale, const unsigned nchanpol, const uint64_t chanpol_stride, const float scale_factor, const uint64_t median_idx)
{
  const unsigned ichanpol = blockIdx.x * blockDim.x + threadIdx.x;
  if (ichanpol >= nchanpol)
  {
    return;
  }

  const uint64_t chanpol_offset = ichanpol * chanpol_stride;
  sorted += chanpol_offset;

  float stddev = sorted[median_idx] * scale_factor;
  if (stddev == 0.0)
  {
    stddev = 1.0;
  }

  scale[ichanpol] = 1.0/stddev;
}

/**
 * @brief CUDA kernel to transpose from elements in TFP order, with the stride of the time samples
 * being istride elements into FPT ordered data.
 *
 * Templating designed to support real data (float) and complex data (float2).
 *
 * @param in input pointer to time series container in TFP order
 * @param out output pointer to bit series container in FPT order, for this kernel FPT is contiguous
 * @param nsamp the number of time samples to handle
 * @param nintegrated the number of time samples already integrated, used as an offset
 * @param ndat number of time samples in the input container
 * @param nchanpol product of the number of channels and number of polarisations
 */
template<typename T>
__global__ void rescale_sample_tfp_to_fpt(const T * __restrict__ in, T * __restrict__ out, unsigned nsamp, uint64_t nintegrated, unsigned nchanpol, uint64_t ndat)
{
  // shared memory structured as [chanpol][dat] use +1 on second dimension to inhibit shared memory bank conflicts
  __shared__ T sdata[32][33];

  const unsigned block_idat = blockIdx.x * blockDim.x;
  const unsigned block_ichanpol = blockIdx.y * blockDim.y;

  const unsigned idat = block_idat + threadIdx.x;
  const unsigned ichanpol = block_ichanpol + threadIdx.y;

  if ((idat < nsamp) && (ichanpol < nchanpol))
  {
    sdata[threadIdx.y][threadIdx.x] = in[idat * nchanpol + ichanpol];
  }
  __syncthreads();

  const unsigned odat = block_idat + threadIdx.y;
  const unsigned ochanpol = block_ichanpol + threadIdx.x;

  if ((odat < nsamp) && (ochanpol < nchanpol))
  {
    out[ochanpol * ndat + nintegrated + odat] = sdata[threadIdx.x][threadIdx.y];
  }
}

void CUDA::RescaleMedianMadCalculatorCUDA::init(const dsp::TimeSeries* input, uint64_t _ndat, bool output_time_total)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::init started" << endl;

  gpu_config.init();

  ndat = _ndat;
  npol = input->get_npol();
  ndim = input->get_ndim();
  nchan = input->get_nchan();
  nchanpol = nchan * npol;

  sample_size_bytes = ndat * nchan * npol * ndim * sizeof(float);
  data_size_bytes = npol * nchan * sizeof(float);
  chanpol_offset_size_bytes = npol * nchan * sizeof(uint64_t);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::init "
      << "ndat=" << ndat
      << ", npol=" << npol
      << ", ndim=" << ndim
      << ", nchan=" << nchan
      << ", sample_size_bytes=" << sample_size_bytes
      << ", data_size_bytes=" << data_size_bytes
      << ", chanpol_offset_size_bytes=" << chanpol_offset_size_bytes
      << endl;

  init_common_memory(data_size_bytes);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::init allocating device ptrs" << endl;

  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_chanpol_start_offset), chanpol_offset_size_bytes);
  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_chanpol_end_offset), chanpol_offset_size_bytes);

  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_sample_data), sample_size_bytes);
  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_absolute_deviation), sample_size_bytes);
  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_sorted), sample_size_bytes);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::init initialising device ptrs." << endl;

  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_chanpol_start_offset), chanpol_offset_size_bytes, stream);
  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_chanpol_end_offset), chanpol_offset_size_bytes, stream);

  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_sample_data), sample_size_bytes, stream);
  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_absolute_deviation), sample_size_bytes, stream);
  RescaleEngine::init_device_memory(reinterpret_cast<void **>(d_sorted), sample_size_bytes, stream);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::init initialising host ptrs" << endl;

  RescaleEngine::allocate_host_memory(reinterpret_cast<void **>(&h_chanpol_start_offset), chanpol_offset_size_bytes);
  RescaleEngine::allocate_host_memory(reinterpret_cast<void **>(&h_chanpol_end_offset), chanpol_offset_size_bytes);
  RescaleEngine::init_host_memory(reinterpret_cast<void *>(h_chanpol_start_offset), chanpol_offset_size_bytes);
  RescaleEngine::init_host_memory(reinterpret_cast<void *>(h_chanpol_end_offset), chanpol_offset_size_bytes);

  for (auto idx = 0; idx < nchanpol; idx++)
  {
    h_chanpol_start_offset[idx] = ndat * ndim * idx;
  }
  cudaError_t error = cudaMemcpyAsync(d_chanpol_start_offset, h_chanpol_start_offset, chanpol_offset_size_bytes, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleMedianMadCalculatorCUDA::init", "cudaMemcpyAsync from h_chanpol_start_offset to d_chanpol_start_offset failed");

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::init setting up host data vectors" << endl;

  median.resize (npol);
  variance.resize (npol);
  scale.resize (npol);
  offset.resize (npol);

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    median[ipol].resize (nchan, 0);
    variance[ipol].resize (nchan, 0);
    scale[ipol].resize (nchan, 0);
    offset[ipol].resize (nchan, 0);
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::init finished" << endl;

}

void CUDA::RescaleMedianMadCalculatorCUDA::reset_sample_data()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::reset_sample_data started" << endl;

  ScaleOffsetCalculatorCUDA::reset_sample_data();

  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_sorted), sample_size_bytes, stream);
  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_sample_data), sample_size_bytes, stream);
  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_absolute_deviation), sample_size_bytes, stream);

  nintegrated = 0;

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::reset_sample_data finished" << endl;
}

auto CUDA::RescaleMedianMadCalculatorCUDA::sample_data(const dsp::TimeSeries* input, uint64_t start_dat, uint64_t end_dat, bool output_time_total) -> uint64_t
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sample_data started "
      << "nintegrated=" << nintegrated
      << ", start_dat=" << start_dat
      << ", end_dat=" << end_dat
      << ", nsamp=" << (end_dat - start_dat) << endl;

  auto nsamp = end_dat - start_dat;

  // ensure we don't integrate too many samples
  if (nintegrated + nsamp > ndat)
    nsamp = ndat - nintegrated;

  if (nsamp == 0)
    return nintegrated;

  switch (input->get_order())
  {
  case dsp::TimeSeries::OrderFPT:
  {
    uint64_t src_offset = start_dat * ndim;
    const float *src = input->get_datptr(0, 0) + src_offset;
    const size_t spitch = input->get_stride() * sizeof(float);
    const size_t width = nsamp * ndim * sizeof(float);

    uint64_t dst_offset = nintegrated * ndim;
    float *dst = d_sample_data + dst_offset;
    const size_t dpitch = ndat * ndim * sizeof(float);
    const size_t height = nchanpol;

    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sample_data FPT performing cudaMemcpy2DAsync"
        << " start_dat=" << start_dat
        << ", end_dat=" << end_dat
        << ", nsamp=" << nsamp
        << ", src_offset=" << src_offset
        << ", nintegrated=" << nintegrated
        << ", dst_offset=" << dst_offset
        << ", input->get_datptr(0, 0)=" << input->get_datptr(0, 0)
        << ", *src=" << src
        << ", d_sample_data=" << d_sample_data
        << ", *dst=" << dst
        << ", dpitch=" << dpitch
        << ", spitch=" << spitch
        << ", width=" << width
        << ", height=" << height
        << endl;

    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);

    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream("CUDA::RescaleMedianMadCalculatorCUDA::sample_data cudaMemcpy2DAsync", stream);

    break;
  }
  case dsp::TimeSeries::OrderTFP:
  default:
  {
    // Note, always used threads(32, max_threads/32, 1).  Previously we limited the number of
    // threads based on NDAT and NCHANPOL but this caused an issue when NCHANPOL < 32
    dim3 threads(32, gpu_config.get_max_threads_per_block() / 32, 1);
    dim3 blocks(nsamp / threads.x, nchanpol / threads.y, 1);
    if (nsamp % threads.x != 0)
    {
      blocks.x++;
    }
    if (nchanpol % threads.y != 0)
    {
      blocks.y++;
    }

    const float *in = input->get_dattfp();

    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sample_data TFP calling rescale_sample_tfp_to_fpt kernel "
        << "blocks=(" << blocks.x << ", " << blocks.y << ", " << blocks.z << ")"
        << ", threads=(" << threads.x << ", " << threads.y << ", " << threads.z << ")"
        << ", nchanpol=" << nchanpol
        << ", nsamp=" << nsamp
        << ", ndat=" << ndat
        << ", ndim=" << ndim
        << ", start_dat=" << start_dat
        << ", in=" << in
        << ", d_sample_data=" << d_sample_data
        << endl;

    if (ndim == 1)
    {
      rescale_sample_tfp_to_fpt<float><<<blocks, threads, 0, stream>>>(
        in + start_dat * nchanpol, d_sample_data, nsamp, nintegrated, nchanpol, ndat
      );
    }
    else
    {
      rescale_sample_tfp_to_fpt<float2><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2 *>(in + start_dat * nchanpol * ndim), reinterpret_cast<float2 *>(d_sample_data),
        nsamp, nintegrated, nchanpol, ndat
      );
    }

    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream("CUDA::RescaleMedianMadCalculatorCUDA::sample_data rescale_sample_tfp_to_fpt", stream);

    break;
  }
  }

  cudaStreamSynchronize(stream);

  nintegrated += nsamp;

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sample_data finished" << endl;

  return nintegrated;
}

void CUDA::RescaleMedianMadCalculatorCUDA::compute()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::compute started - nintegrated=" << nintegrated << endl;

  auto chanpol_stride = ndat * ndim;

  // this is to be consistent with CPU version - if ndat is odd the idx would be the middle value
  // if ndat is even, such NDAT=4, we would get the idx of 1 (if we had 1-offset indexing then it would be 2)
  const uint64_t median_idx = (nintegrated * ndim - 1) / 2;

  // the start offsets can be calculated in init as they don't change
  for (auto idx = 0; idx < nchanpol; idx++)
  {
    h_chanpol_end_offset[idx] = h_chanpol_start_offset[idx] + nintegrated * ndim;
  }
  cudaError_t error = cudaMemcpyAsync(d_chanpol_end_offset, h_chanpol_end_offset, chanpol_offset_size_bytes, cudaMemcpyHostToDevice, stream);

  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleMedianMadCalculatorCUDA::compute", "cudaMemcpyAsync from h_chanpol_end_offset to d_chanpol_end_offset failed");

  // sort data to allow getting median
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::compute sorting sampled data" << endl;

  sort_data(d_sample_data);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::compute calling rescale_calculate_offset kernel - "
      << "ndat=" << (nintegrated * ndim)
      << ", chanpol_stride=" << chanpol_stride
      << ", median_idx=" << median_idx
      << endl;

  // kernel to perform calculating the offset
  rescale_calculate_offset<<<nchanpol, nthreads, 0, stream>>>(d_sorted, d_offset, d_absolute_deviation, nintegrated * ndim, chanpol_stride, median_idx);
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::RescaleMedianMadCalculatorCUDA::compute rescale_calculate_offset", stream);

  // sort absolute deviation to allow getting MAD
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::compute sorting absolute deviation from median" << endl;

  sort_data(d_absolute_deviation);

  // kernel to select get the scale
  uint64_t numblocks = nchanpol / nthreads;
  if (nchanpol % nthreads)
  {
    numblocks++;
  }
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::compute calling rescale_calculate_scale kernel - "
      << "nchanpol=" << nchanpol
      << ", chanpol_stride=" << chanpol_stride
      << ", numblocks=" << numblocks
      << ", median_idx=" << median_idx
      << endl;

  rescale_calculate_scale<<<numblocks, nchanpol, 0, stream>>>(d_sorted, d_scale, nchanpol, chanpol_stride, 1.0/scale_factor, median_idx);
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::RescaleMedianMadCalculatorCUDA::compute rescale_calculate_scale", stream);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::compute finished" << endl;
}

void CUDA::RescaleMedianMadCalculatorCUDA::transfer_to_host()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::transfer_to_host started" << endl;

  cudaError_t error;

  error = cudaMemcpyAsync(h_offset, d_offset, data_size_bytes, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleMedianMadCalculatorCUDA::transfer_to_host", "cudaMemcpyAsync from d_offset to h_offset failed");

  error = cudaMemcpyAsync(h_scale, d_scale, data_size_bytes, cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleMedianMadCalculatorCUDA::transfer_to_host", "cudaMemcpyAsync from d_scale to h_scale failed");

  check_error_stream("CUDA::RescaleMedianMadCalculatorCUDA::transfer_to_host stream sync", stream);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::transfer_to_host nchan=" << nchan << ", npol=" << npol << endl;

  // Need to transpose from FP to PF and store in std::vectors
  for (auto ichan = 0; ichan < nchan; ichan++)
  {
    for (auto ipol = 0; ipol < npol; ipol++)
    {
      auto ichanpol = ichan * npol + ipol;
      auto curr_offset = h_offset[ichanpol];
      auto curr_scale = h_scale[ichanpol];

      if (dsp::Operation::verbose)
        cerr << "CUDA::RescaleMedianMadCalculatorCUDA::transfer_to_host ichan=" << ichan << ", ipol="
          << ipol << ", ichanpol=" << ichanpol << ", curr_offset=" << curr_offset << ", curr_scale=" << curr_scale << endl;

      offset[ipol][ichan] = curr_offset;
      scale[ipol][ichan] = curr_scale;
      median[ipol][ichan] = static_cast<double>(-curr_offset);
      if (curr_scale != 0.0)
        variance[ipol][ichan] = 1.0 / static_cast<double>(curr_scale * curr_scale);
    }
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::transfer_to_host finished" << endl;
}

void CUDA::RescaleMedianMadCalculatorCUDA::sort_data(float* input_data)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sort_data started" << endl;

  size_t required_temp_storage_size_bytes = 0;

  auto num_segments = nchanpol;
  auto num_items = ndat * ndim * num_segments;

  // calling SortKeys with nullptr for the temp storage will set the
  // required_temp_storage_size_bytes
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sort_data calling SortKey with "
      << "d_temp_storage=" << d_temp_storage
      << ", temp_storage_size_bytes=" << required_temp_storage_size_bytes
      << ", input_data=" << input_data
      << ", d_sorted=" << d_sorted
      << ", num_items=" << num_items
      << ", num_segments=" << num_segments
      << ", d_chanpol_start_offset" << d_chanpol_start_offset
      << ", d_chanpol_end_offset" << d_chanpol_end_offset
      << endl;

  cub::DeviceSegmentedSort::SortKeys(
    nullptr, required_temp_storage_size_bytes, input_data, d_sorted,
    num_items, num_segments, d_chanpol_start_offset, d_chanpol_end_offset, stream);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::RescaleMedianMadCalculatorCUDA::sort_data stream sync - get required_temp_storage_size_bytes failed", stream);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sort_data required_temp_storage_size_bytes=" << required_temp_storage_size_bytes << endl;

  if (required_temp_storage_size_bytes > temp_storage_size_bytes)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sort_data "
        << "required_temp_storage_size_bytes=" << required_temp_storage_size_bytes
        << " larger than current temp_storage_size_bytes=" << temp_storage_size_bytes << endl;

    if (temp_storage_size_bytes)
    {
      if (dsp::Operation::verbose)
        cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sort_data "
          << "releasing device memory of d_temp_storage" << endl;

      RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_temp_storage));
    }

    RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_temp_storage), required_temp_storage_size_bytes);
    temp_storage_size_bytes = required_temp_storage_size_bytes;

    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sort_data "
        << "d_temp_storage has been allocated and initialised" << endl;
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sort_data calling SortKey with "
      << "d_temp_storage=" << d_temp_storage
      << ", temp_storage_size_bytes=" << temp_storage_size_bytes
      << ", input_data=" << input_data
      << ", d_sorted=" << d_sorted
      << ", num_items=" << num_items
      << ", num_segments=" << num_segments
      << ", d_chanpol_start_offset" << d_chanpol_start_offset
      << ", d_chanpol_end_offset" << d_chanpol_end_offset
      << endl;

  cub::DeviceSegmentedSort::SortKeys(
    d_temp_storage, temp_storage_size_bytes, input_data, d_sorted,
    num_items, num_segments, d_chanpol_start_offset, d_chanpol_end_offset, stream);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::RescaleMedianMadCalculatorCUDA::sort_data stream sync - sort keys failed", stream);

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleMedianMadCalculatorCUDA::sort_data finished" << endl;
}
