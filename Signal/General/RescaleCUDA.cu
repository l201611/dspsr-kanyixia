//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <assert.h>

#include "dsp/RescaleCUDA.h"
#include "dsp/RescaleMeanStdDevCalculatorCUDA.h"

#include <cstring>

#define FULLMASK 0xFFFFFFFF

using namespace std;

static constexpr unsigned nthreads = 1024;

void check_error_stream(const char *, cudaStream_t);

CUDA::RescaleEngine::RescaleEngine(cudaStream_t _stream) : stream(_stream)
{
  calculator = new CUDA::RescaleMeanStdDevCalculatorCUDA(_stream);
}

CUDA::RescaleEngine::~RescaleEngine()
{
}

void CUDA::RescaleEngine::allocate_device_memory(void ** device_ptr, size_t nbytes)
{
  if (*device_ptr != nullptr)
    throw Error(InvalidParam, "CUDA::RescaleEngine::allocate_device_memory",
      "device_ptr expected to be a nullptr, perhaps device_ptr has already been allocated");

  // allocate and set device memory
  cudaError_t error = cudaMalloc(device_ptr, nbytes);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleEngine::allocate_device_memory",
      "cudaMalloc for %u bytes failed: %s", nbytes, cudaGetErrorString(error));
}

void CUDA::RescaleEngine::init_device_memory(void *device_ptr, size_t nbytes, cudaStream_t _stream)
{
  if (device_ptr == nullptr)
    throw Error(InvalidParam, "CUDA::RescaleEngine::init_device_memory",
      "device_ptr was a nullptr, expected it to have already been allocated");

  cudaError_t error = cudaMemsetAsync(device_ptr, 0, nbytes, _stream);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleEngine::init_device_memory",
      "cudaMemsetAsync for %u bytes failed: %s", nbytes, cudaGetErrorString(error));
}

void CUDA::RescaleEngine::release_device_memory(void **device_ptr)
{
  if (*device_ptr == nullptr)
    return;

  cudaError_t error = cudaFree(*device_ptr);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleEngine::release_device_memory",
      "cudaFree failed: %s", cudaGetErrorString(error));

  *device_ptr = nullptr;
}

void CUDA::RescaleEngine::allocate_host_memory(void ** host_ptr, size_t nbytes)
{
  if (*host_ptr != nullptr)
    throw Error(InvalidParam, "CUDA::RescaleEngine::allocate_host_memory",
      "host_ptr expected to be a nullptr, perhaps host_ptr has already been allocated");

  cudaError_t error = cudaMallocHost(reinterpret_cast<void **>(host_ptr), nbytes);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleEngine::allocate_host_memory",
      "cudaMallocHost for %u bytes failed: %s", nbytes, cudaGetErrorString(error));
}

void CUDA::RescaleEngine::release_host_memory(void ** host_ptr)
{
  cudaError_t error = cudaFreeHost(*host_ptr);
  if (error != cudaSuccess)
    throw Error(FailedCall, "CUDA::RescaleEngine::release_host_memory",
      "cudaFreeHost failed: %s", cudaGetErrorString(error));

  *host_ptr = nullptr;
}

void CUDA::RescaleEngine::init_host_memory(void * host_ptr, size_t nbytes)
{
  if (host_ptr == nullptr)
    throw Error(InvalidParam, "CUDA::RescaleEngine::init_host_memory",
      "host_ptr was a nullptr, expected it to have already been allocated");

  std::memset(host_ptr, 0, nbytes);
}

/*
  Apply the current offset and scales to a TFP-ordered Timeseries and write it to an output TFP-ordered Timeseries.

  @param in base address of TimeSeries that should be rescaled.
  @param out the base address of the TimeSeries that rescaled values should be written to.
  @param offset the base address of where the computed offsets are stored in FP order.
  @param scale the base address of where the computed scales are stored in FP order.
  @param nchan the number of channels that the input timeseries has.
  @param npol the number of polarizations that the input timeseries has.
  @param ndim the number of dimensions the samples in the input timeseries has.

  Kernel assumes each CUDA block processes exactly 1 time sample.

  out[idx] = (in[idx] + offset[ichanpol]) * scale[ichanpol]
*/
__global__ void rescale_apply_offset_scale_tfp(const float *in,
                                               float *out,
                                               const float *offset, const float *scale,
                                               const unsigned nchan, const unsigned npol, const unsigned ndim)
{
  // each block will process 1 sample
  const uint64_t dat_offset = blockIdx.x * nchan * npol * ndim;

  // process all the channels in this block
  for (unsigned ichan = threadIdx.x; ichan < nchan; ichan += blockDim.x)
  {
    // input and output offsets
    uint64_t idx = dat_offset + (ichan * npol * ndim);

    for (unsigned ipol = 0; ipol < npol; ipol++)
    {
      const unsigned ichanpol = ichan * npol + ipol;
      const float ichanpol_offset = offset[ichanpol];
      const float ichanpol_scale = scale[ichanpol];
      for (unsigned idim = 0; idim < ndim; idim++)
      {
        // convert to 0 mean and unit variance
        out[idx] = (in[idx] + ichanpol_offset) * ichanpol_scale;
        idx++;
      }
    }
  }
}

/*
  Apply the current offset and scales to a FPT-ordered Timeseries and write it to an output FPT-ordered Timeseries.

  Calculate the offsets, scales, freq total and freq squared totals for FPT-ordered data.

  @param in base address of TimeSeries that should be rescaled.
  @param out the base address of the TimeSeries that rescaled values should be written to.
  @param chanpol_stride the stride in data between a channel and a polarisation.
  @param offset the base address of where the computed offsets are stored in FP order.
  @param start_dat which time sample to start from, usually 0 but may not be.
  @param ndat the total number of time samples to be used to calculate the statistics.
  @param ndim the number of dimensions the samples in the input timeseries has.

  Kernel assumes each CUDA block processes exactly 1 channel/polarisation combination.

  out[idx] = (in[idx] + offset[ichanpol]) * scale[ichanpol]

*/
__global__ void rescale_apply_offset_scale_fpt(const float *in, float *out,
                                               uint64_t chanpol_stride,
                                               const float *offset, const float *scale,
                                               uint64_t start_dat, uint64_t ndat, unsigned ndim)
{
  const unsigned ichanpol = blockIdx.x;
  const uint64_t chanpol_offset = ichanpol * chanpol_stride;

  const float *in_ptr = in + chanpol_offset;
  float *out_ptr = out + chanpol_offset;

  const float ichanpol_offset = offset[ichanpol];
  const float ichanpol_scale = scale[ichanpol];

  for (uint64_t idat = threadIdx.x; idat < ndat; idat += blockDim.x)
  {
    uint64_t idat_idx = (idat + start_dat) * ndim;
    for (unsigned idim = 0; idim < ndim; idim++)
    {
      out_ptr[idat_idx + idim] = (in_ptr[idat_idx + idim] + ichanpol_offset) * ichanpol_scale;
    }
  }
}

void CUDA::RescaleEngine::rescale(const dsp::TimeSeries* input, dsp::TimeSeries* output, uint64_t start_dat, uint64_t end_dat)
{
  auto nchanpol = nchan * npol;
  auto nsamp = end_dat - start_dat;

  const float* d_offset = calculator->get_device_offsets();
  const float* d_scale = calculator->get_device_scales();

  switch (input->get_order())
  {
  case dsp::TimeSeries::OrderTFP:
  {
    const float *in = input->get_dattfp();
    uint64_t ptr_offset = start_dat * nchanpol * ndim;
    unsigned nblocks = nsamp;

    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleEngine::rescale calling rescale_apply_offset_scale_tfp nblocks=" << nblocks
            << ", nthreads=" << nthreads << ", ptr_offset=" << ptr_offset << endl;

    rescale_apply_offset_scale_tfp<<<nblocks, nthreads, 0, stream>>>(in + ptr_offset, output->get_dattfp() + ptr_offset,
                                                                      d_offset, d_scale, nchan, npol, ndim);

    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream("CUDA::RescaleEngine::rescale rescale_apply_offset_scale_tfp", stream);

    break;
  }
  case dsp::TimeSeries::OrderFPT:
  {
    unsigned nblocks = nchanpol;
    auto chanpol_stride = input->get_stride();

    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleEngine::rescale calling rescale_apply_offset_scale_fpt nblocks=" << nblocks
            << ", nthreads=" << nthreads << ", chanpol_stride=" << chanpol_stride << endl;

    rescale_apply_offset_scale_fpt<<<nblocks, nthreads, 0, stream>>>(input->get_datptr(0, 0), output->get_datptr(0, 0),
                                                                      chanpol_stride,
                                                                      d_offset, d_scale,
                                                                      start_dat, nsamp, ndim);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream("CUDA::RescaleEngine::rescale rescale_apply_offset_scale_fpt", stream);

    break;
  }
  }
}

void CUDA::RescaleEngine::init(const dsp::TimeSeries *input, uint64_t _nsample, bool _exact, bool _constant_offset_scale)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleEngine::init started" << endl;

  gpu_config.init();

  nsample = _nsample;
  exact = _exact;
  constant_offset_scale = _constant_offset_scale;

  npol = input->get_npol();
  ndim = input->get_ndim();
  nchan = input->get_nchan();
  nchanpol = nchan * npol;

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleEngine::init calling calculator->init()" << endl;

  calculator->init(input, _nsample, false);

  first_integration = true;

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleEngine::init finished" << endl;
}

void CUDA::RescaleEngine::transform(const dsp::TimeSeries *input, dsp::TimeSeries *output)
{
  const auto input_ndat = exact ? nsample : input->get_ndat();

  uint64_t start_dat = 0;
  uint64_t end_dat = 0;

  // only perform the rescale if we're constant and have done the first integration
  if (constant_offset_scale && !first_integration)
  {
    rescale(input, output, 0, input_ndat);
    return;
  }

  while (start_dat < input_ndat)
  {
    // default to sampling until the end
    auto nsamp = input_ndat - start_dat;

    // don't sample more than needed to calculate stats
    if (isample + nsamp > nsample)
      nsamp = nsample - isample;

    end_dat = start_dat + nsamp;

    if (dsp::Operation::verbose)
      cerr << "CUDA::RescaleEngine::transform "
        << "start_dat=" << start_dat
        << ", end_dat=" << end_dat
        << ", nsamp=" << nsamp
        << ", isample=" << isample
        << endl;

    if (first_integration || !constant_offset_scale)
      isample = calculator->sample_data(input, start_dat, end_dat, false);

    // perform calculation of scales and offsets if:
    // * we have sampled enough time samples to perform a calculation
    // * or, we are at the end of the input data on the first integration
    if ((isample == nsample) || (first_integration && end_dat == input_ndat))
    {
      calculator->compute();
      calculator->transfer_to_host();

      // ensure we have the correct time when scales and offsets were updated
      if (update_epoch == MJD::zero)
        update_epoch = input->get_start_time();
      update_epoch += isample / input->get_rate();

      fire_scales_updated(input, start_dat);

      if (isample == nsample)
      {
        first_integration = false;
        isample = 0;
        calculator->reset_sample_data();
      }
    }

    // this will allow breaking out of loop but also to ensure we
    // apply rescaling to the rest of the input data.
    if (constant_offset_scale && !first_integration)
      end_dat = input_ndat;

    rescale(input, output, start_dat, end_dat);
    start_dat = end_dat;
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::RescaleEngine::transform exiting" << endl;
}

const float *CUDA::RescaleEngine::get_offset(unsigned ipol) const
{
  return calculator->get_offset(ipol);
}

const float *CUDA::RescaleEngine::get_scale(unsigned ipol) const
{
  return calculator->get_scale(ipol);
}

const double *CUDA::RescaleEngine::get_freq_total(unsigned ipol) const
{
  return calculator->get_mean(ipol);
}

const double *CUDA::RescaleEngine::get_freq_squared_total(unsigned ipol) const
{
  return calculator->get_variance(ipol);
}

void CUDA::RescaleEngine::set_calculator(dsp::Rescale::ScaleOffsetCalculator* _calculator)
{
  auto cuda_calculator = dynamic_cast<ScaleOffsetCalculatorCUDA *>(_calculator);
  if (!cuda_calculator)
    throw Error (InvalidParam, "CUDA::RescaleEngine::set_calculator",
                "expected calculator to be a pointer to a subclass of CUDA::ScaleOffsetCalculatorCUDA");

  calculator = cuda_calculator;
}

void CUDA::RescaleEngine::fire_scales_updated(const dsp::TimeSeries* input, uint64_t start_dat)
{
  dsp::ASCIIObservation observation(input);

  auto sample_offset = static_cast<uint64_t>(input->get_input_sample()) + start_dat;

  dsp::Rescale::update_record record{
    sample_offset,
    // the number of values used to determine the scales and offsets include all dimensions per sample
    isample * static_cast<uint64_t>(input->get_ndim()),
    calculator->get_scales(),
    calculator->get_offsets(),
    observation
  };
  scales_updated(record);
}

MJD CUDA::RescaleEngine::get_update_epoch () const
{
  return update_epoch;
}

CUDA::ScaleOffsetCalculatorCUDA::ScaleOffsetCalculatorCUDA(cudaStream_t _stream) : stream(_stream), dsp::Rescale::ScaleOffsetCalculator()
{
}

CUDA::ScaleOffsetCalculatorCUDA::~ScaleOffsetCalculatorCUDA()
{
  RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_scale));
  RescaleEngine::release_device_memory(reinterpret_cast<void **>(&d_offset));

  RescaleEngine::release_host_memory(reinterpret_cast<void **>(&h_scale));
  RescaleEngine::release_host_memory(reinterpret_cast<void **>(&h_offset));
}

void CUDA::ScaleOffsetCalculatorCUDA::init_common_memory(size_t data_size_bytes)
{
  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_scale), data_size_bytes);
  RescaleEngine::allocate_device_memory(reinterpret_cast<void **>(&d_offset), data_size_bytes);

  RescaleEngine::allocate_host_memory(reinterpret_cast<void **>(&h_scale), data_size_bytes);
  RescaleEngine::allocate_host_memory(reinterpret_cast<void **>(&h_offset), data_size_bytes);

  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_scale), data_size_bytes, stream);
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::ScaleOffsetCalculatorCUDA::init_common_memory - initialising d_scale", stream);

  RescaleEngine::init_host_memory(reinterpret_cast<void *>(h_scale), data_size_bytes);

  RescaleEngine::init_device_memory(reinterpret_cast<void *>(d_offset), data_size_bytes, stream);
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::ScaleOffsetCalculatorCUDA::init_common_memory - initialising d_offset", stream);

  RescaleEngine::init_host_memory(reinterpret_cast<void *>(h_offset), data_size_bytes);
}

void CUDA::ScaleOffsetCalculatorCUDA::reset_sample_data() {}
