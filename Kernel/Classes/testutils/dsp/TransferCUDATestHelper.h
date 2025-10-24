/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <dsp/TransferCUDA.h>

#ifndef __dsp_TransferCUDATestHelper_h
#define __dsp_TransferCUDATestHelper_h

#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDA.h"
#include "dsp/TransferBitSeriesCUDA.h"

namespace dsp::test {

/**
 * @brief Facilitates the transfer of TimeSeries to/from GPU
 *
 */
class TransferCUDATestHelper
{
  public:

    /**
     * @brief Construct a new TransferCUDATestHelper object.
     *
     * Note that the memory attribute must be CUDA::DeviceMemory instance if a host to device
     * transfer is used by the helper.
     *
     * @param memory optional memory manager that will allocate memory on device
     * @param stream optional CUDA stream in which transfers will be scheduled
     */
    TransferCUDATestHelper(Memory* memory = 0, cudaStream_t stream = 0);

    /**
     * @brief Destroy the TransferCUDATestHelper object
     *
     */
    ~TransferCUDATestHelper() = default;

    /**
     * @brief Copy data from one TimeSeries to another, either from device to host or vice versa
     *
     * @param into TimeSeries into which data will be copied
     * @param from TimeSeries from which data will be copied
     * @param direction the kind of transfer (device to host, or host to device)
     * @post If direction == cudaMemcpyHostToDevice and this->memory has been set,
     *       then the memory manager of out will be set to CUDA::DeviceMemory
     */
    void copy(dsp::TimeSeries* into, const dsp::TimeSeries* from, cudaMemcpyKind direction);

    /**
     * @brief Copy data from one BitSeries to a another, either from device to host or vice versa
     *
     * @param into BitSeries into which data will be copied
     * @param from BitSeries from which data will be copied
     * @param direction the kind of transfer (device to host, or host to device)
     * @post If direction == cudaMemcpyHostToDevice and this->memory has been set,
     *       then the memory manager of out will be set to CUDA::DeviceMemory
     */
    void copy_bitseries(dsp::BitSeries* into, const dsp::BitSeries* from, cudaMemcpyKind direction);

  private:

    /**
     * @brief Check that the memory manager is castable to a CUDA::DeviceMemory pointer
     */
    void assert_gpu_memory();

    Reference::To<dsp::TransferCUDA> transfer;
    Reference::To<dsp::TransferBitSeriesCUDA> transfer_bitseries;
    Reference::To<Memory> memory;
};

} // namespace dsp::test

#endif // __dsp_TransferCUDATestHelper_h
