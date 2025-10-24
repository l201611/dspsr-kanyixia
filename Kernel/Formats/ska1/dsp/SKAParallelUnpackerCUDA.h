//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024 by Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Kernel_Formats_ska1_cuda_SKAParallelUnpacker_h
#define __dsp_Kernel_Formats_ska1_cuda_SKAParallelUnpacker_h

#include "dsp/SKAParallelUnpacker.h"

#include <cuda_runtime.h>

namespace CUDA {

  class SKAParallelUnpackerEngine : public dsp::SKAParallelUnpacker::Engine
  {
    public:

      //! Default Constructor
      SKAParallelUnpackerEngine (cudaStream_t stream = 0);

      //! Perform any internal setup
      void setup (dsp::SKAParallelUnpacker* user) override;

      /**
       * @brief Unpack the ParallelBitSeries to the output TimeSeries, both in GPU memory.
       *
       * @param data BitSeries containing the data
       * @param weights BitSeries containing the scales and weights
       * @param output TimeSeries to which the unpacked data will be written
       * @param nsamp_per_packet number of time samples per UDP packet
       * @param nchan_per_packet number of channels per UDP packet
       * @param nsamp_per_weight number of time samples spanned by each relative weight
       * @param weights_valid flag that indicates the weights are valid and should be respected
       */
      void unpack (const dsp::BitSeries * data, const dsp::BitSeries * weights, dsp::TimeSeries * output, uint32_t nsamp_per_packet, uint32_t nchan_per_packet, uint32_t nsamp_per_weight, bool weights_vaild) override;

    protected:

      //! CUDA stream in which this transformation will operate
      cudaStream_t stream;

      //! gpu configuration
      struct cudaDeviceProp gpu;
  };
} // namespace dsp

#endif // __dsp_Kernel_Formats_ska1_cuda_SKAParallelUnpacker_h
