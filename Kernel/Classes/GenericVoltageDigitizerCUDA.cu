//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024 by Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericVoltageDigitizerCUDA.h"
#include "dsp/TFPOffset.h"

#include "Error.h"

#include <assert.h>
#include <stdio.h>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

//! CUDA implementation of a quantization algorithm for use by the GenericVoltageDigitizer
CUDA::GenericVoltageDigitizerEngine::GenericVoltageDigitizerEngine (cudaStream_t _stream)
{
  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::GenericVoltageDigitizerEngine constructor stream=" << reinterpret_cast<void*>(_stream) << endl;
  }
  stream = _stream;
  scratch_space = nullptr;
  scratch_space_size = 0;
}

void CUDA::GenericVoltageDigitizerEngine::setup (dsp::GenericVoltageDigitizer* user)
{
  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::GenericVoltageDigitizerEngine::setup()" << endl;
  }
  dsp::GenericVoltageDigitizer::Engine::setup(user);
  gpu_config.init();
  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::GenericVoltageDigitizerEngine::setup gpu_config.get_max_threads_per_block()="
         << gpu_config.get_max_threads_per_block() << endl;
  }
}

void CUDA::GenericVoltageDigitizerEngine::set_scratch_space (void * _scratch_space, size_t _scratch_space_size)
{
  scratch_space = reinterpret_cast<float2 *>(_scratch_space);
  scratch_space_size = _scratch_space_size;
}

/**
 * @brief CUDA kernel to transpose from elements in FPT order, with the stride of the time samples
 * being istride elements into TFP ordered data. The kernels uses shared memory to transpose
 * 32 time samples from 32 chanpols.
 *
 * Templating designed to support real data (float, uint16) and complex data (float2)
 *
 * @param in input pointer to time series container in FPT order
 * @param out output pointer to bit series container in TFP order
 * @param ndat number of time samples in the input container
 * @param nchanpol product of the number of channels and number of polarisations
 */
template<typename T>
__global__ void generic_voltage_digitizer_transpose_fpt_tfp(const T * __restrict__ in, T * __restrict__ out, unsigned istride, unsigned ndat, unsigned nchanpol)
{
  // shared memory structured as [chanpol][dat] use +1 on second dimension to inhibant shared memory bank conflicts
  __shared__ T sdata[32][33];

  const unsigned block_idat = blockIdx.x * blockDim.x;
  const unsigned block_ichanpol = blockIdx.y * blockDim.y;

  // each block transpose 32 time and up to 32 chanpols
  const unsigned idat = block_idat + threadIdx.x;
  const unsigned ichanpol = block_ichanpol + threadIdx.y;

  if ((idat < ndat) && (ichanpol < nchanpol))
  {
    sdata[threadIdx.y][threadIdx.x] = in[ichanpol * istride + idat];
  }
  __syncthreads();

  const unsigned odat = block_idat + threadIdx.y;
  const unsigned ochanpol = block_ichanpol + threadIdx.x;

  if ((odat < ndat) && (ochanpol < nchanpol))
  {
    out[odat * nchanpol + ochanpol] = sdata[threadIdx.x][threadIdx.y];
  }
}

/**
 * @brief CUDA kernel for digitizing TFP-ordered input floats to 1 bit per sample
 *
 * @param input_ptr input pointer to time series container in TFP order
 * @param output_ptr output pointer to bit series container in TFP order
 * @param nfloat number of floats to pack inside 1 byte
 */
__global__ void generic_voltage_digitizer_pack_1b_tfp(
  const float * input_ptr,
  char * output_ptr,
  uint64_t nfloat
)
{
  unsigned nfloats_per_thread = 8;
  const unsigned char mask = 0x01;
  const unsigned thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  // Return if the product of the thread index and floats per thread exceeds the total number of floats.
  if (thread_index * nfloats_per_thread >= nfloat)
  {
    return;
  }

  // Each thread goes through nfloats_per_thread at a time
  input_ptr += thread_index * nfloats_per_thread;

  char outval = 0;
  for (int ifloat=0; ifloat <nfloats_per_thread; ifloat++)
  {
    int result = signbit(input_ptr[ifloat]) ? 0 : 1;
    outval |= ((static_cast<char>(result) & mask) <<  ifloat);
  }
  output_ptr[thread_index] = outval;
}

/**
 * @brief CUDA kernel for digitizing TFP-ordered input floats to 2 or 4 bits per sample
 *
 * @param input_ptr input pointer to time series container in TFP order
 * @param output_ptr output pointer to bit series container in TFP order
 * @param nfloat total number of floats to pack
 * @param effective_scale factor by which each float is multiplied before it is cast to an integer
 * @param digi_mean value subtracted from each float before it is cast to an integer
 * @param digi_min the minimum allowed value of digitized values
 * @param digi_max the maximum allowed value of digitized values
 * @param nbit the number of bits per output sample
 */
__global__ void generic_voltage_digitizer_pack_2b_and_4b_tfp(
  const float * input_ptr,
  char * output_ptr,
  uint64_t nfloat,
  float effective_scale,
  float digi_mean,
  int digi_min,
  int digi_max,
  int nbit
)
{
  unsigned nfloats_per_thread = 8 / nbit;
  const unsigned char mask = static_cast<unsigned char>((1 << nbit) - 1);

  const unsigned thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_index * nfloats_per_thread >= nfloat)
    return;

  input_ptr += thread_index * nfloats_per_thread; // each thread goes through nfloats_per_thread at a time

  char outval = 0;
  for (int ifloat=0; ifloat <nfloats_per_thread; ifloat++)
  {
    int scaled = roundf(fmaf(input_ptr[ifloat], effective_scale, digi_mean));
    int result = static_cast<int>(min(max(scaled, digi_min), digi_max));
    outval |= ((static_cast<char>(result) & mask) <<  nbit * ifloat);
  }

  output_ptr[thread_index] = outval;
}

/**
 * @brief CUDA kernel for digitizing TFP-ordered input floats to 8 or 16 bits per sample
 *
 * @param input_ptr input pointer to time series container in TFP order
 * @param output_ptr output pointer to bit series container in TFP order
 * @param nfloat total number of floats to pack
 * @param effective_scale factor by which each float is multiplied before it is cast to an integer
 * @param digi_mean value subtracted from each float before it is cast to an integer
 * @param digi_min the minimum allowed value of digitized values
 * @param digi_max the maximum allowed value of digitized values
*/
template<typename T>
__global__ void generic_voltage_digitizer_pack_8b_and_16b_tfp(
  const float * input_ptr,
  T * output_ptr,
  uint64_t nfloat,
  float effective_scale,
  float digi_mean,
  int digi_min,
  int digi_max
)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nfloat)
    return;

  int scaled = roundf(fmaf(input_ptr[idx], effective_scale, digi_mean));
  int result = static_cast<int>(min(max(scaled, digi_min), digi_max));
  output_ptr[idx] = static_cast<T>(result);
}

/**
 * @brief CUDA kernel for digitizing to 32 bit TFP ordered input container
 *
 * @param input_ptr input pointer to time series container in TFP order
 * @param output_ptr output pointer to bit series container in TFP order
 * @param nfloat total number of floats to pack
 * @param effective_scale factor by which each float is multiplied
*/
__global__ void generic_voltage_digitizer_pack_32b_tfp(
  const float* input_ptr,
  float* output_ptr,
  uint64_t nfloat,
  float effective_scale
)
{
  const uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= nfloat)
    return;

  output_ptr[idx] = input_ptr[idx] * effective_scale;
}

void CUDA::GenericVoltageDigitizerEngine::pack (
  const dsp::TimeSeries * input,
  dsp::BitSeries * output,
  const int nbit,
  const int digi_min,
  const int digi_max,
  const float digi_mean,
  const float digi_scale,
  const float effective_scale,
  const int samp_per_byte
)
{
  // the number of frequency channels
  unsigned nchan = input->get_nchan();
  // the number of time samples
  uint64_t ndat = input->get_ndat();
  // number of polarizations
  unsigned npol = input->get_npol();
  // number of dimensions
  unsigned ndim = input->get_ndim();

  const float * tfp_input = nullptr;

  // TFP ordered input requires no transpose
  if (input->get_order() == dsp::TimeSeries::OrderTFP)
  {
    tfp_input = reinterpret_cast<const float *>(input->get_dattfp());
  }
  // a transpose from FPT to TFP is required
  else
  {
    size_t input_size_bytes = input->internal_get_size();

    // check that the provided scratch space is sufficient
    size_t required_scratch_space = nchan * ndat * npol * ndim * sizeof(float);
    if (required_scratch_space > scratch_space_size)
    {
      throw Error(InvalidState, "CUDA::GenericVoltageDigitizerEngine::pack",
        "required_scratch_space_size=%u is greater than scratch_space_size=%u", required_scratch_space, scratch_space_size);
    }

    // input stride in float2 elements
    uint32_t input_stride = input->get_stride() / ndim;

    unsigned nchanpol = nchan * npol;

    // Note, always used threads(32, max_threads/32, 1).  Previously we limited the number of
    // threads based on NDAT and NCHANPOL but this caused an issue when NCHANPOL < 32
    dim3 threads(32, gpu_config.get_max_threads_per_block() / 32, 1);
    dim3 blocks(ndat / threads.x, nchanpol / threads.y, 1);
    if (ndat % threads.x != 0)
    {
      blocks.x++;
    }
    if (nchanpol % threads.y != 0)
    {
      blocks.y++;
    }

    if (dsp::Operation::verbose)
    {
      cerr << "CUDA::GenericVoltageDigitizerEngine::pack nchan=" << nchan << " npol=" << npol << " ndat=" << ndat << endl;
      cerr << "CUDA::GenericVoltageDigitizerEngine::pack input=" << input->get_datptr(0, 0) << " scratch_space=" << scratch_space << endl;
      cerr << "CUDA::GenericVoltageDigitizerEngine::pack blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << "), "
           << "threads=(" << threads.x << "," << threads.y << "," << threads.z << ")" << endl;
    }

    generic_voltage_digitizer_transpose_fpt_tfp<float2><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const float2 *>(input->get_datptr(0, 0)),
      scratch_space, input_stride, ndat, nchanpol
    );

    if (dsp::Operation::record_time || dsp::Operation::verbose)
    {
      check_error_stream("CUDA::GenericVoltageDigitizerEngine::pack generic_voltage_digitizer_transpose_fpt_tfp", stream);
    }

    tfp_input = reinterpret_cast<const float *>(scratch_space);
  }

  // perform packing from TFP ordered input to TFP ordered output, these kernels treat all
  // values as sequential
  uint64_t nfloat = ndat * nchan * npol * ndim;
  uint64_t nfloats_per_thread = 1;

  if (nbit < 8 && nbit >= 1)
  {
    nfloats_per_thread = 8/nbit; // 8 = bits per byte
    auto remainder = nfloat % nfloats_per_thread;
    if (remainder != 0)
    {
      nfloat += (nfloats_per_thread - remainder);
    }
  }

  const int total_nthreads = nfloat / nfloats_per_thread;
  unsigned nthreads = gpu_config.get_max_threads_per_block();
  unsigned nblocks = total_nthreads / nthreads;
  if (total_nthreads % nthreads)
  {
    nblocks++;
  }

  if (dsp::Operation::verbose)
  {
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::pack total_nthreads=" << total_nthreads
              << " nblocks=" << nblocks << " nthreads=" << nthreads << " nbit=" << nbit << std::endl;
  }

  const float * input_ptr = reinterpret_cast<const float *>(tfp_input);

  switch (nbit)
  {
    case 1:
    {
      char * output_ptr = reinterpret_cast<char *>(output->get_rawptr());
      generic_voltage_digitizer_pack_1b_tfp<<<nblocks,nthreads,0,stream>>> (
        input_ptr,
        output_ptr,
        nfloat
      );
      break;
    }
    case 2:
    case 4:
    {
      char * output_ptr = reinterpret_cast<char *>(output->get_rawptr());
      generic_voltage_digitizer_pack_2b_and_4b_tfp<<<nblocks,nthreads,0,stream>>> (
        input_ptr,
        output_ptr,
        nfloat,
        effective_scale,
        digi_mean,
        digi_min,
        digi_max,
        nbit
      );
      break;
    }
    case 8:
    {
      int8_t * output_ptr = reinterpret_cast<int8_t *>(output->get_rawptr());
      generic_voltage_digitizer_pack_8b_and_16b_tfp<<<nblocks,nthreads,0,stream>>> (
        input_ptr,
        output_ptr,
        nfloat,
        effective_scale,
        digi_mean,
        digi_min,
        digi_max
      );
      break;
    }
    case 16:
    {
      int16_t * output_ptr = reinterpret_cast<int16_t *>(output->get_rawptr());
      generic_voltage_digitizer_pack_8b_and_16b_tfp<<<nblocks,nthreads,0,stream>>> (
        input_ptr,
        output_ptr,
        nfloat,
        effective_scale,
        digi_mean,
        digi_min,
        digi_max
      );
      break;
    }
    case -32:
    {
      float * output_ptr = reinterpret_cast<float *>(output->get_rawptr());
      generic_voltage_digitizer_pack_32b_tfp<<<nblocks,nthreads,0,stream>>> (
        input_ptr,
        output_ptr,
        nfloat,
        effective_scale
      );
      break;
    }
    default:
    {
      throw Error(InvalidState, "CUDA::GenericVoltageDigitizerEngine::pack",
                  "Unrecognized number of bits per output sample.");
    }
    if (dsp::Operation::verbose)
    {
      std::cerr << "CUDA::GenericVoltageDigitizerEngine::pack generic_voltage_digitizer_pack_*_tfp completed" << std::endl;
    }
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
  {
    check_error_stream ("CUDA::GenericVoltageDigitizerEngine::pack generic_voltage_digitizer_pack_Nb", stream);
  }
}

void CUDA::GenericVoltageDigitizerEngine::copy_weights(const dsp::WeightedTimeSeries * weighted_input, dsp::BitSeries * output_weights)
{
  if (!weighted_input || !output_weights)
    return;

  if (dsp::Operation::verbose)
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights called - "
      << "weighted_input=" << weighted_input
      << " output_weights=" << output_weights
      << std::endl;

  const auto nchan_weight = weighted_input->get_nchan_weight();
  const auto npol_weight = weighted_input->get_npol_weight();
  const auto nchanpol_weight = nchan_weight * npol_weight;
  const auto nweights = weighted_input->get_nweights();
  const auto ndat_per_weight = weighted_input->get_ndat_per_weight();

  // Ensure the correct configuration and size
  output_weights->copy_configuration(weighted_input);
  output_weights->set_nchan(nchan_weight);
  output_weights->set_npol(npol_weight);

  // Weights are 16-bit and real-valued
  output_weights->set_nbit(16);
  output_weights->set_ndim(1);

  // Weights have a reduced sampling rate
  output_weights->set_rate(weighted_input->get_rate() / ndat_per_weight);

  if (dsp::Operation::verbose)
  {
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights resizing output_weights for nweights=" << nweights << std::endl;
  }
  output_weights->resize (nweights);

  if (dsp::Operation::verbose)
  {
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights nchan_weight: " << nchan_weight << std::endl;
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights npol_weight: " << npol_weight << std::endl;
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights nchanpol_weight: " << nchanpol_weight << std::endl;
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights nweights: " << nweights << std::endl;
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights ndat_per_weight: " << ndat_per_weight << std::endl;
  }

  const unsigned input_stride = weighted_input->get_weights_stride();

  // Note, always used threads(32, max_threads/32, 1).  Previously we limited the number of
  // threads based on NDAT and NCHANPOL but this caused an issue when NCHANPOL < 32
  dim3 threads(32, gpu_config.get_max_threads_per_block() / 32, 1);
  dim3 blocks(nweights / threads.x, nchanpol_weight / threads.y, 1);
  if (nweights % threads.x != 0)
  {
    blocks.x++;
  }
  if (nchanpol_weight % threads.y != 0)
  {
    blocks.y++;
  }

  const auto in = weighted_input->get_weights(0, 0);
  auto out = reinterpret_cast<uint16_t*>(output_weights->get_rawptr());

  if (dsp::Operation::verbose)
  {
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights transposing from FPT to TFP" << std::endl;
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights in=" << in << " out=" << out << std::endl;
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << "), "
          << "threads=(" << threads.x << "," << threads.y << "," << threads.z << ")" << std::endl;
  }

  generic_voltage_digitizer_transpose_fpt_tfp<uint16_t><<<blocks, threads, 0, stream>>>(
    in, out, input_stride, nweights, nchanpol_weight
  );

  if (dsp::Operation::record_time || dsp::Operation::verbose)
  {
    check_error_stream("CUDA::GenericVoltageDigitizerEngine::copy_weights generic_voltage_digitizer_transpose_fpt_tfp", stream);
  }

  if (dsp::Operation::verbose)
    std::cerr << "CUDA::GenericVoltageDigitizerEngine::copy_weights transpose and copy complete" << std::endl;

}
