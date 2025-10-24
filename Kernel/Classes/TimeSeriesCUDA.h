//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2016-2025 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_cuda_TimeSeries_h
#define __dsp_cuda_TimeSeries_h

#include "dsp/LaunchConfig.h"
#include "dsp/MemoryCUDA.h"
#include "dsp/TimeSeries.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class TimeSeriesEngine : public dsp::TimeSeries::Engine
  {
  public:

    //! Default constructor
    TimeSeriesEngine (dsp::Memory * _memory);

    /**
     * @brief Destroy the Time Series Engine object.
     *
     * Releases any memory allocated to the buffer or host_buffer
     */
    ~TimeSeriesEngine ();

    /**
     * @brief Set the Timeseries to which this engine will copy data.
     *
     * @param parent destination time series to copy data to
     */
    void prepare (dsp::TimeSeries * parent);

    /**
     * @brief Ensure the buffer of device memory is at least nbytes in size.
     *
     * @param nbytes required size of the buffer in bytes
     */
    void prepare_buffer (unsigned nbytes);

    /**
     * @brief Copy data from another time series to this time series, assuming FPT ordering.
     *
     * @param from other time series from which to copy data
     * @param idat_start first sample in the remote timeseries
     * @param ndat number of time samples to copy
     */
    void copy_data_fpt (const dsp::TimeSeries * from,
                        uint64_t idat_start = 0,
                        uint64_t ndat = 0);

    //! pointer to a device memory buffer for the time series data
    void* buffer = nullptr;

  protected:

   /**
     * @brief Copy FPT ordered data from another time series when both timeseries
     * are using the same CUDA device and stream.
     *
     * @param from other time series from which to copy data
     * @param idat_start first sample in the remote timeseries
     * @param ndat number of time samples to copy
     * @param nchanpol the number of channels * polarisations
     */
    void copy_data_fpt_same_stream (const dsp::TimeSeries * from,
            uint64_t idat_start, uint64_t ndat, unsigned nchanpol);

    /**
     * @brief Copy FPT ordered data from another time series when both timeseries
     * are using the same CUDA device, but different CUDA streams.
     *
     * @param from other time series from which to copy data
     * @param idat_start first sample in the remote timeseries
     * @param ndat number of time samples to copy
     * @param nchanpol the number of channels * polarisations
     */
    void copy_data_fpt_same_device (const dsp::TimeSeries * from,
            uint64_t idat_start, uint64_t ndat, unsigned nchanpol);

    /**
     * @brief Copy FPT ordered data from another time series when both timeseries
     * are using different CUDA devices and streams.
     *
     * @param from other time series from which to copy data
     * @param idat_start first sample in the remote timeseries
     * @param ndat number of time samples to copy
     * @param nchanpol the number of channels * polarisations
     */
    void copy_data_fpt_diff_device (const dsp::TimeSeries * from,
            uint64_t idat_start, uint64_t ndat, unsigned nchanpol);

    /**
     * @brief Copy FPT ordered data between device memory buffers.
     *
     * @param to destination device memory buffer
     * @param from source device memory buffer
     * @param to_stride number of elements (float or float2) between each chanpol in the destination buffer
     * @param from_stride number of elements (float or float2) between each chanpol in the source buffer
     * @param idat_start first sample in the source timeseries
     * @param ndat number of samples to copy from the source timeseries
     * @param nchanpol the number of channels * polarisations
     * @param stream CUDA stream in which to perform the copy
     */
    void copy_data_fpt_kernel_multidim (float * to, const float * from,
            uint64_t to_stride, uint64_t from_stride,
            uint64_t idat_start, uint64_t ndat, unsigned nchanpol, cudaStream_t stream);

    //! the destination time series to which data will be copied
    dsp::TimeSeries* to = nullptr;

    //! CUDA device memory manager
    CUDA::DeviceMemory* memory = nullptr;

    //! Pinned host memory manager
    CUDA::PinnedMemory* pinned_memory = nullptr;

    //! CUDA configuration assistant
    CUDA::LaunchConfig gpu_config;

    //! pointer to host memory that is used when transferring between devices
    void* host_buffer = nullptr;

    //! size of the host memory buffer in bytes
    size_t host_buffer_size = 0;

    //! size of the device memory buffer in bytes
    size_t buffer_size = 0;

    //! number of channels
    unsigned nchan = 0;

    //! number of polarisations
    unsigned npol = 0;

    //! number of dimensions
    unsigned ndim = 0;

    //! number of elements (float or float2) between each chanpol in the source/from buffer
    uint64_t ichanpol_stride = 0;

    //! number of elements (float or float2) between each chanpol in the destination/to buffer
    uint64_t ochanpol_stride = 0;

    //! number of elements (float or float2) between each chanpol in the temporary buffer
    uint64_t bchanpol_stride = 0;

    //! number of threads in a CUDA block
    unsigned nthread = 0;

    //! size of the CUDA grid in X, Y and Z dimensions
    dim3 blocks;

    //! the CUDA device in the context that calls copy_data_fpt
    int device = -1;

    //! CUDA stream used by the destination timeseries
    cudaStream_t to_stream;

    //! CUDA stream used by the source timeseries
    cudaStream_t from_stream;

    //! CUDA device used by the destination timeseries
    int to_device = -1;

    //! CUDA device used by the source timeseries
    int from_device = -1;

  };

} // namespace CUDA

#endif // !defined(__dsp_cuda_TimeSeries_h)
