//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ChanPolSelectCUDA.h"
#include "dsp/TFPOffset.h"
#include "dsp/WeightedTimeSeries.h"

#include "Error.h"
#include "debug.h"

#include <cassert>

using namespace std;

void check_error_stream (const char*, cudaStream_t stream);

CUDA::ChanPolSelectEngine::ChanPolSelectEngine (cudaStream_t _stream)
{
  stream = _stream;
}

//! get cuda device properties
void CUDA::ChanPolSelectEngine::setup(dsp::ChanPolSelect* user)
{
  dsp::ChanPolSelect::Engine::setup(user);
  gpu_config.init();
}

/*
  Copy FPT-ordered data from one TimeSeries to another, also used to copy
  FPT-ordered weights from one WeightedTimeSeries to another.

  @param input base address from where data are copied
  @param output base address to where data are copied
  @param in_chan_stride number of values between consecutive input channels
  @param in_pol_stride number of values between consecutive input polarizations
  @param in_chan_stride number of values between consecutive output channels
  @param in_pol_stride number of values between consecutive output polarizations
  @param nval total number of values to be copied for each channel and polarization

  each thread copies a single value from input to output

  blockIdx.x * blockDim.x + threadIdx.x = index of the float to be copied
  blockIdx.y = index of the frequency channel to be copied
  blockIdx.z = index of the polarization to be copied
*/
template <typename T>
__global__ void fpt_copy
(
  const T * __restrict__ input,
  T * output,
  uint64_t in_chan_stride,
  uint64_t in_pol_stride,
  uint64_t out_chan_stride,
  uint64_t out_pol_stride,
  uint64_t nval)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nval)
    return;

  // increment the input/output base pointers to this chan/pol
  input += blockIdx.y * in_chan_stride + blockIdx.z * in_pol_stride;
  output += blockIdx.y * out_chan_stride + blockIdx.z * out_pol_stride;

  output[idx] = input[idx];
}

/*
  Copy TFP-ordered data from one TimeSeries to another

  @param input base address of TimeSeries from where data are copied
  @param output base address of TimeSeries to where data are copied
  @param input_offset computes offset from input base address for (idat, ichan, ipol)
  @param output_offset computes offset from output base address for (idat, ichan, ipol)
  @param nfloat total number of floats to be copied for each channel and polarization
  @param ndim the number of floats copied for each time sample

  nfloat = ndat * ndim
  where ndat is the number of time samples to be copied for each channel and polarization

  each thread copies a single float from input to output

  blockIdx.x * blockDim.x + threadIdx.x = index of the float to be copied
  blockIdx.y = index of the frequency channel to be copied
  blockIdx.z = index of the polarization to be copied

  This implementation is not optimal - it should probably be re-implemented with
  blockDim.x = nchan * npol * ndim
*/
__global__ void tfp_copy
(
  const float * __restrict__ input,
  float * output,
  dsp::TFPOffset input_offset,
  dsp::TFPOffset output_offset,
  uint64_t nfloat,
  unsigned ndim)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nfloat)
    return;

  unsigned idat = idx / ndim;
  unsigned idim = idx % ndim;

  auto input_index = input_offset(idat, blockIdx.y, blockIdx.z);
  auto output_index = output_offset(idat, blockIdx.y, blockIdx.z);

  output[output_index + idim] = input[input_index + idim];
}

//! Copies the selected frequency channels and polarizations for FPT ordered data
void CUDA::ChanPolSelectEngine::select (const dsp::TimeSeries* input, dsp::TimeSeries* output)
{
  assert(input != nullptr);
  assert(output != nullptr);

  if (input == output)
    throw Error (InvalidParam, "CUDA::ChanPolSelectEngine::select", "cannot handle in-place data");

  const uint64_t ndat = output->get_ndat();
  assert(input->get_ndat() == ndat);

  const unsigned ndim = output->get_ndim();
  assert(input->get_ndim() == ndim);

  const unsigned output_nchan = output->get_nchan();
  assert(number_of_channels_to_keep == output_nchan);

  const unsigned output_npol = output->get_npol();
  assert(number_of_polarizations_to_keep == output_npol);

  const uint64_t nfloat = ndat * ndim;

  unsigned nthreads = std::min(nfloat, gpu_config.get_max_threads_per_block());
  dim3 blocks (nfloat / nthreads, output_nchan, output_npol);
  if (nfloat % nthreads)
    blocks.x++;

  switch (input->get_order())
  {
    case dsp::TimeSeries::OrderTFP:
    {
      dsp::TFPOffset input_offset(input);
      dsp::TFPOffset output_offset(output);

      if (dsp::Operation::verbose)
        std::cerr << "CUDA::ChanPolSelectEngine::select get input base TFP pointer" << std::endl;

      const float* in = input->get_dattfp();
      in += input_offset(0, start_channel_index, start_polarization_index);

      if (dsp::Operation::verbose)
        std::cerr << "CUDA::ChanPolSelectEngine::select get output base TFP pointer" << std::endl;
      float* out = output->get_dattfp();

      tfp_copy<<<blocks,nthreads,0,stream>>> (in, out, input_offset, output_offset, nfloat, ndim);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream("CUDA::ChanPolSelectEngine::select tfp_copy data", stream);
    }
    break;

    case dsp::TimeSeries::OrderFPT:
    {
      uint64_t in_pol_stride = input->get_stride();
      uint64_t in_chan_stride = in_pol_stride * input->get_npol();

      uint64_t out_pol_stride = output->get_stride();
      uint64_t out_chan_stride = out_pol_stride * output_npol;

      const float* in = input->get_datptr(start_channel_index, start_polarization_index);
      float* out = output->get_datptr(0, 0);

      if (dsp::Operation::verbose)
        std::cerr << "CUDA::ChanPolSelectEngine::select nfloat=" << nfloat << " out_pol_stride=" << out_pol_stride << " in_pol_stride=" << in_pol_stride << endl;

      fpt_copy<<<blocks,nthreads,0,stream>>> (in, out, in_chan_stride, in_pol_stride, out_chan_stride, out_pol_stride, nfloat);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream("CUDA::ChanPolSelectEngine::select fpt_copy data", stream);
    }
    break;
  }

  // if the input and output time series are WeightedTimeSeries, and the input has populated weights
  auto weighted_output = dynamic_cast<dsp::WeightedTimeSeries*>(output);
  auto weighted_input = dynamic_cast<const dsp::WeightedTimeSeries*>(input);
  if (weighted_output && weighted_input && weighted_input->get_ndat_per_weight() > 0)
  {
    if (dsp::Operation::verbose)
      std::cerr << "CUDA::ChanPolSelectEngine::select copying weights into WeightedTimeSeries" << std::endl;

    uint64_t in_pol_stride = weighted_input->get_weights_stride();
    uint64_t in_chan_stride = in_pol_stride * weighted_input->get_npol_weight();

    uint64_t out_pol_stride = weighted_output->get_weights_stride();
    uint64_t out_chan_stride = out_pol_stride * weighted_output->get_npol_weight();

    const uint16_t* in = weighted_input->get_weights(start_channel_index, start_polarization_index);
    uint16_t* out = weighted_output->get_weights(0, 0);

    uint64_t nweights = weighted_input->get_nweights();
    nthreads = std::min(nweights, gpu_config.get_max_threads_per_block());
    blocks.x = nweights / nthreads;
    if (nweights % nthreads != 0)
      blocks.x++;

    if (dsp::Operation::verbose)
    {
      std::cerr << "CUDA::ChanPolSelectEngine::select weights blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << ") nthreads=" << nthreads << std::endl;
      std::cerr << "CUDA::ChanPolSelectEngine::select weights in_chan_stride=" << in_chan_stride << " in_pol_stride=" << in_pol_stride << " out_chan_stride=" << out_chan_stride << " out_pol_stride=" << out_pol_stride << std::endl;
    }
    fpt_copy<<<blocks,nthreads,0,stream>>> (in, out, in_chan_stride, in_pol_stride, out_chan_stride, out_pol_stride, nweights);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream("CUDA::ChanPolSelectEngine::select fpt_copy weights", stream);
  }
}
