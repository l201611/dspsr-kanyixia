//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024 by Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_cuda_GenericVoltageDigitizer_h
#define __dsp_cuda_GenericVoltageDigitizer_h

#include "dsp/GenericVoltageDigitizer.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/LaunchConfig.h"
#include "dsp/Scratch.h"

namespace CUDA
{
  //! CUDA implementation of the GenericVoltageDigitizer.
  class GenericVoltageDigitizerEngine : public dsp::GenericVoltageDigitizer::Engine
  {
    public:

      /**
       * @brief Construct a new GenericVoltageDigitizerEngine object.
       *
       * @param stream CUDA stream in which GPU operations will be scheduled.
       */
      GenericVoltageDigitizerEngine (cudaStream_t stream = 0);

      /**
       * @brief Perform internal setup operations for the engine.
       *
       * @param user pointer to the transformation instance that uses this engine
       */
      void setup (dsp::GenericVoltageDigitizer* user) override;

      /**
       * @brief Set the scratch space to be used by the engine.
       *
       * @param scratch_space pointer to GPU memory usable as scratch space
       * @param scratch_space_size size of the scratch space in bytes
       */
      void set_scratch_space (void * scratch_space, size_t scratch_space_size);

      /**
      * @brief Digitize the samples in the input TimeSeries, writing the quantized
      * representation to the output BitSeries at the specified bit depth.
      *
      * @param input input data TimeSeries
      * @param output input data BitSeries
      * @param nbit number of bits per datum
      * @param digi_min the minimum allowed value of digitized values
      * @param digi_max the maximum allowed value of digitized values
      * @param digi_mean the mean offset the digitized output.
      * @param digi_scale the scale/standard deviation of the digitized output.
      * @param effective_scale factor by which each float is multiplied before it is cast to an integer
      * @param samp_per_byte number of samples per byte
      */
      void pack (
        const dsp::TimeSeries * input,
        dsp::BitSeries * output,
        const int nbit,
        const int digi_min,
        const int digi_max,
        const float digi_mean,
        const float digi_scale,
        const float effective_scale,
        const int samp_per_byte
      ) override;

      /**
       * @brief Copy weights from input WeightedTimeSeries to output BitSeries.
       * Transposes the order from FPT WeightedTimeSeries to TFP BitSeries.
       *
       * @param weighted_input the input WeightedTimeSeries
       * @param output_weights the output BitSeries
       */
      void copy_weights(const dsp::WeightedTimeSeries * weighted_input, dsp::BitSeries * output_weights) override;
    protected:

      //! CUDA stream in which this transformation will operate
      cudaStream_t stream = 0;

      //! CUDA configuration assistant
      LaunchConfig gpu_config;

      //! scratch used for handling of transpose GPU memory
      float2* scratch_space = nullptr;

      //! size of the scratch space in bytes
      size_t scratch_space_size = 0;
  };

} // namespace CUDA

#endif // __dsp_cuda_GenericVoltageDigitizer_h
