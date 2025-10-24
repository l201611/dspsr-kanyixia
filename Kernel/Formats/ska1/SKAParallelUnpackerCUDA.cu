//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKAParallelUnpackerCUDA.h"
#include "dsp/WeightedTimeSeries.h"

#include "Error.h"
#include <stdio.h>
#include <cfloat>     // for MIN_FLT

using namespace std;

void check_error_stream (const char*, cudaStream_t);

CUDA::SKAParallelUnpackerEngine::SKAParallelUnpackerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::SKAParallelUnpackerEngine::setup (dsp::SKAParallelUnpacker* user)
{
  dsp::SKAParallelUnpacker::Engine::setup(user);
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

/**
 * @brief CUDA kernel for unpacking data and weights bitseries container into a timeseries container
 *
 * The TimeSeries container is structured in Frequency, Polarisation, and Time (FPT) order.
 *
 * @param data input data bitseries container
 * @param weights input scale+weights bitseries container
 * @param output_pol0 pointer to pol 0 of the output container
 * @param output_pol1 pointer to pol 1 of the output container
 * @param heap_stride size in bytes of a heap per packet
 * @param data_packet_stride size (in number of bytes) of data per packet
 * @param weights_packet_stride offset (in number of uint16_t) between scale in each packet
 * @param ochan_stride offset (in number of floats) between output frequency channels
 *
 * Each Packet is ordered as FPT
 * For Low that with be 24 channels, 2 pols, 32 samples, 2 dims **
 * For Mid that will be 185 channels, 2 pols, 4 samples, 2 dims (AA0.5 and AA1)
 * For Mid that will be 1 channels, 2 pols, 1024 samples, 2 dims (AA2+)
 *
 * nchan_per_packet == blockDim.y
 * nsamp_per_packet == blockDim.x
 * ipacket == blockIdx.x
 * iheap == blockIdx.y
 * channel within a packet == threadIdx.y
 * sample within a packet == threadIdx.x
 *
 */
template <typename T>
__global__ void ska_parallel_unpack_fpt(
  const T* data, const short* scales_weights,
  float2* output_pol0, float2* output_pol1,
  uint32_t heap_stride, uint32_t data_packet_stride, uint32_t weights_packet_stride, uint32_t ochan_stride)
{
  // input offset for ichan, idat and pol=0
  //                         iheap * heap_stride  +     ipacket * data_packet_stride  +  packet_chan * pack_nsamp * 2  + packet_sample
  const uint32_t idx = (blockIdx.y * heap_stride) + (blockIdx.x * data_packet_stride) + (threadIdx.y * blockDim.x * 2) + threadIdx.x;
  T data_pol_0 = data[idx];
  T data_pol_1 = data[idx + blockDim.x];

  // we need to get the correct scale from the weights bit series
  // since the Mid CBF format can have 185 x uint16_t weights between the float scale factors, some trickery is required to ensure
  // the float read correctly and properly aligned
  scales_weights += (blockIdx.x * weights_packet_stride);
  short2 scale_parts = { scales_weights[0], scales_weights[1] };
  float * scale = reinterpret_cast<float *>(&scale_parts);
  const float scale_factor = *scale;

  // unpack the data by dividing by the scale_factor
  const float multiplier = (isnan(scale_factor) || scale_factor<FLT_MIN) ? 0.0 : 1.0 / scale_factor;

  // write out pol0 and pol1
  const uint32_t ichan = (blockIdx.x * blockDim.y) + threadIdx.y;
  const uint32_t idat = (blockIdx.y * blockDim.x) + threadIdx.x;
  const uint32_t odx = (ichan * ochan_stride);
  output_pol0[odx + idat] = make_float2(float(data_pol_0.x) * multiplier, float(data_pol_0.y) * multiplier);
  output_pol1[odx + idat] = make_float2(float(data_pol_1.x) * multiplier, float(data_pol_1.y) * multiplier);
}

/**
 * @brief CUDA kernel for unpacking weights bitseries container into a weighted timeseries weights array
 *
 * The WeightedTimeSeries weights array is structured in Frequency, Polarisation, and Time (FPT) order.
 *
 * @param input_weights base address of input BitSeries scale+weights array
 * @param output_weights base address of output WeightedTimeSeries weight array
 * @param npackets_per_heap number of packets per heap
 * @param nchan_per_packet number of frequency channels per packet
 * @param input_weights_stride offset (in number of 16-bit integers) between weights in each packet
 * @param output_weights_stride offset (in number of 16-bit integers) between output frequency channels
 * @param weights_valid that indicates the weights are valid and should be respected
 *
 */
__global__ void ska_parallel_unpack_weights(
  const uint16_t* input_weights, uint16_t* output_weights, uint32_t npackets_per_heap, uint32_t nchan_per_packet, int32_t input_weights_stride, uint32_t output_weights_stride, bool weights_valid
  )
{
  const uint32_t ichan = threadIdx.x;
  const uint32_t ipacket = blockIdx.x;
  const uint32_t iheap = blockIdx.y;

  // output offset for packet and channel
  output_weights += (ipacket * nchan_per_packet + ichan) * output_weights_stride;

  if (weights_valid)
  {
    // input offset for heap and packet
    input_weights += (iheap * npackets_per_heap + ipacket) * input_weights_stride;

    output_weights[iheap] = input_weights[ichan];
  }
  else
  {
    output_weights[iheap] = 1.0;
  }
}

void CUDA::SKAParallelUnpackerEngine::unpack (const dsp::BitSeries * data, const dsp::BitSeries * weights, dsp::TimeSeries * output, uint32_t nsamp_per_packet, uint32_t nchan_per_packet, uint32_t nsamp_per_weight, bool weights_valid)
{
  const uint64_t ndat = data->get_ndat();

  if (ndat == 0)
    return;

  const unsigned nchan = data->get_nchan();
  const unsigned ndim = data->get_ndim();
  const unsigned npol = data->get_npol();
  const unsigned nbit = data->get_nbit();

  assert (ndim == 2);
  assert (npol == 2);
  assert (nbit == 8 || nbit == 16);

  const uint32_t npol_per_packet = 2;
  const uint32_t weights_nbyte = 2;

  const uint32_t nheaps = ndat / nsamp_per_packet;
  const uint32_t npackets_per_heap = nchan / nchan_per_packet;

  const void* data_from = reinterpret_cast<const void*>(data->get_rawptr());
  const short* scales_weights_from = reinterpret_cast<const short*>(weights->get_rawptr());

  float2* into_pol0 = reinterpret_cast<float2 *>(output->get_datptr(0, 0));
  float2* into_pol1 = reinterpret_cast<float2 *>(output->get_datptr(0, 1));

  uint32_t data_packet_stride = nsamp_per_packet * nchan_per_packet * npol_per_packet;
  uint32_t heap_stride = npackets_per_heap * data_packet_stride;

  uint32_t scale_stride_bytes = sizeof(float);
  uint32_t weights_bytes = ((nchan_per_packet * weights_nbyte * nsamp_per_packet) / nsamp_per_weight);
  uint32_t weights_packet_stride = (scale_stride_bytes + weights_bytes) / sizeof(short);
  uint32_t out_chan_stride = (output->get_datptr(1, 0) - output->get_datptr(0, 0)) / ndim;

  if (output->get_order() != dsp::TimeSeries::OrderFPT)
    throw Error (InvalidState, "CUDA::SKAParallelUnpackerEngine::unpack",
                 "can only unpack into FPT order");

  dim3 threads = dim3(nsamp_per_packet, nchan_per_packet, 1); // 768 threads
  dim3 blocks = dim3(npackets_per_heap, nheaps, 1);

  if (threads.x * threads.y * threads.z > gpu.maxThreadsPerBlock)
  {
    throw Error(InvalidParam, "CUDA::SKAParallelUnpackerEngine::unpack", "Block dimensions (%u, %u, %u) > maxThreadsPerBlock=%u", threads.x, threads.y, threads.z, gpu.maxThreadsPerBlock);
  }

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::SKAParallelUnpackerEngine::unpack ndat=" << ndat << " nchan=" << nchan << " ndim=" << ndim <<" npol=" << npol << " nbit=" << nbit << endl;
    cerr << "CUDA::SKAParallelUnpackerEngine::unpack nsamp_per_packet=" << nsamp_per_packet << " nchan_per_packet=" << nchan_per_packet << " nsamp_per_weight=" << nsamp_per_weight << std::endl;
    cerr << "CUDA::SKAParallelUnpackerEngine::unpack blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << ") threads=(" << threads.x << "," << threads.y << "," << threads.z << ")" << std::endl;
    cerr << "CUDA::SKAParallelUnpackerEngine::unpack heap_stride=" << heap_stride << " data_packet_stride=" << data_packet_stride << " weights_packet_stride=" << weights_packet_stride << " out_chan_stride=" << out_chan_stride << std::endl;
    cerr << "CUDA::SKAParallelUnpackerEngine::unpack data=" << data_from << " scales_weights=" << scales_weights_from << " into_pol0=" << into_pol0 << " into_pol1=" << into_pol1 << endl;
  }

  if (nbit == 16)
    ska_parallel_unpack_fpt<short2><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const short2*>(data_from), scales_weights_from,
      into_pol0, into_pol1,
      heap_stride, data_packet_stride, weights_packet_stride, out_chan_stride
    );
  else
    ska_parallel_unpack_fpt<char2><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const char2*>(data_from), scales_weights_from,
      into_pol0, into_pol1,
      heap_stride, data_packet_stride, weights_packet_stride, out_chan_stride
    );

  auto weighted_output = dynamic_cast<dsp::WeightedTimeSeries*>(output);

  if (weighted_output)
  {
    //! Get the weights array for the specified polarization and frequency
    uint16_t* output_weights = weighted_output->get_weights ();
    uint32_t output_weights_stride = weighted_output->get_weights_stride ();

    // convert stride in bytes to number of uint16_t weights
    uint32_t input_weights_stride = (scale_stride_bytes + weights_bytes) / sizeof(uint16_t);

    // weights start after the scale in each block
    const uint16_t* input_weights = reinterpret_cast<const uint16_t*>(weights->get_rawptr() + scale_stride_bytes);

    dim3 threads = dim3(nchan_per_packet, 1, 1);
    dim3 blocks = dim3(npackets_per_heap, nheaps, 1);

    if (dsp::Operation::verbose)
    {
      cerr << "CUDA::SKAParallelUnpackerEngine::unpack weights blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << ") threads=(" << threads.x << "," << threads.y << "," << threads.z << ")" << std::endl;
      cerr << "CUDA::SKAParallelUnpackerEngine::unpack input_weights_stride=" << input_weights_stride << " output_weights_stride=" << output_weights_stride << std::endl;
    }

    ska_parallel_unpack_weights<<<blocks, threads, 0, stream>>>(
      input_weights, output_weights,
      npackets_per_heap, nchan_per_packet,
      input_weights_stride, output_weights_stride,
      weights_valid
    );
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::SKAParallelUnpackerEngine::unpack", stream);
  else
    cudaStreamSynchronize(stream);
}
