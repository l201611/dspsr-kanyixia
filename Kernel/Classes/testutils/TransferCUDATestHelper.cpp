
/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferCUDATestHelper.h"

dsp::test::TransferCUDATestHelper::TransferCUDATestHelper(Memory* _memory, cudaStream_t stream)
{
  transfer = new dsp::TransferCUDA (stream);
  transfer_bitseries = new dsp::TransferBitSeriesCUDA (stream);
  memory = _memory;
}

void dsp::test::TransferCUDATestHelper::assert_gpu_memory()
{
  CUDA::DeviceMemory * cuda_device_memory = dynamic_cast<CUDA::DeviceMemory *>(memory.get());
  if (!cuda_device_memory)
  {
    throw Error(InvalidParam, "dsp::test::TransferCUDATestHelper::assert_gpu_memory",
      "memory is not a CUDA::DeviceMemory manager");
  }
}

void dsp::test::TransferCUDATestHelper::copy(dsp::TimeSeries* output, const dsp::TimeSeries* input, cudaMemcpyKind direction)
{
  transfer->set_kind(direction);

  if (memory && direction == cudaMemcpyHostToDevice)
  {
    assert_gpu_memory();
    output->set_memory(memory);
  }

  transfer->set_input(input);
  transfer->set_output(output);
  transfer->operate();
}

void dsp::test::TransferCUDATestHelper::copy_bitseries(dsp::BitSeries* output, const dsp::BitSeries* input, cudaMemcpyKind direction)
{
  transfer_bitseries->set_kind(direction);

  if (memory && direction == cudaMemcpyHostToDevice)
  {
    assert_gpu_memory();
    output->set_memory(memory);
  }

  transfer_bitseries->set_input(input);
  transfer_bitseries->set_output(output);
  transfer_bitseries->operate();
}
