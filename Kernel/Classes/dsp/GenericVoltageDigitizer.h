//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/sigproc/dsp/GenericVoltageDigitizer.h

#ifndef __dsp_GenericVoltageDigitizer_h
#define __dsp_GenericVoltageDigitizer_h

#include "dsp/Digitizer.h"

namespace dsp
{
  class WeightedTimeSeries;

  //! Digitizer that converts FTP or TFP ordered TimeSeries to TFP-ordered, Twos Complement BitSeries.
  class GenericVoltageDigitizer : public Digitizer
  {
  public:

    //! Default constructor
    GenericVoltageDigitizer();

    //! Get the number of bits per sample (FITS BITPIX convention)
    void set_nbit(int) override;

    //! Initialize metadata and resize the outputs
    void prepare () override;

    /**
     * @brief pack the data.
     *
     * This method must transpose the data from frequency major order to
     * time major order.  It is assumed that ndat > 4 * nchan, and therefore
     * stride in output time is smaller than stride in input frequency.
     *
     * For NBIT < 8 the packing of the output data is done in little endian
     * and that the earlier time samples are written in to the least significant
     * bits (LSBs) and for NDIM > 1 the dimensions are written adjacent with the
     * order of the dimensions are also packed from LSBs to most significant
     * (e.g for NDIM=2 the real value is in the lower bits than the imaginary value).
     */
    void pack() override;

    /**
     * @brief Set the memory manager the digitizer will use.
     *
     * If the memory manager is castable to CUDA::DeviceMemory, this method will
     * extract the CUDA stream from the memory manager and use it to:
     *
     *  1. If engine is not set, instantiate a new CUDA::GenericVoltageDigitizerEngine instance
     *  2. Set the engine using the new instance
     *  3. Instantiate a new Scratch instance that uses the same memory manager
     *  4. Set the scratch space to be used by the instance.
     *
     * If the engine is set, this method will setup the engine.
     *
     * @param memory memory manager to use in this Transformation
     */
    void set_device (Memory* memory) override;

    //! Engine used to perform digitization, instead of the default CPU implementation
    class Engine;

    /**
     * @brief Set the engine instance to be used.
     *
     * @param engine pointer to the engine that will be used by this transformation.
     */
    void set_engine (Engine* engine);

    //! Return true if the digitizer can operate on the specified memory
    bool get_device_supported (Memory* memory) const override;

    /**
     * @brief Sets a multiplicative scale factor to apply in addition to the digi_scale.
     *
     * @param scale an additional scale factor to apply when performing digitization.
     */
    void set_scale(float scale) { scale_fac = scale; }

    /**
     * @brief Prepare meta-data for the output weights.
     *
     */
    void prepare_weights();

    /**
     * @brief Copy weights from input WeightedTimeSeries to output BitSeries.
     * Transposes the order from FPT WeightedTimeSeries to TFP BitSeries.
     */
    void copy_weights();

    /**
     * @brief Set the output weights BitSeries instance to be used for storing weights
     *
     * @param _bitseries_weights pointer to the BitSeries that will be used for storing weights
     */
    void set_output_weights(dsp::BitSeries* _bitseries_weights);

    /**
     * @brief True if input TimeSeries has valid weights and output_weights BitSeries is set
     * @pre The prepare method has been called
     *
     */
    bool has_output_weights() const { return weighted_input != nullptr; }

    /**
     * @brief Get the default digitization mean for the requantization bit width
     *
     * @param bit_width number of bits per sample
     * @return float default digitization mean for the requantization bit width
     */
    static float get_default_digi_mean(int bit_width);

    /**
     * @brief Get the default digitization scale factor for the requantization bit width
     *
     * @param bit_width number of bits per sample
     * @return float default digitization scale factor for the requantization bit width
     */
    static float get_default_digi_scale(int bit_width);

  protected:

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! Additional scale factor to apply to data during digitization
    float scale_fac = 1.0;

    //! Special case for output = floating point
    void pack_float();

  private:

    //! output BitSeries weights container
    Reference::To<dsp::BitSeries> output_weights;

    //! interface to WeightedTimeSeries input
    const WeightedTimeSeries* weighted_input = nullptr;

    //! the mean offset the digitized output.
    float digi_mean{0.0};

    //! the scale/standard deviation of the digitized output.
    float digi_scale{0.0};

    //! the maximum allowed value of digitized values
    int digi_max{0};

    //! the minimum allowed value of digitized values
    int digi_min{0};

    //! flag representing the initialisation of a device
    bool device_prepared;
  };


  //! Alternative implementation of the GenericVoltageDigitizer.
  class GenericVoltageDigitizer::Engine : public Reference::Able
  {
    public:

    /**
     * @brief Pack interface of the engine that converts the input TimeSeries floating-point data to bits in a BitSeries.
     *
     * @param input input data TimeSeries
     * @param output input data BitSeries
     * @param nbit number of bits per datum
     * @param digi_min the minimum allowed value of digitized values
     * @param digi_max the maximum allowed value of digitized values
     * @param digi_mean the mean offset the digitized output
     * @param digi_scale the scale/standard deviation of the digitized output
     * @param effective_scale factor by which each float is multiplied before it is cast to an integer
     * @param samp_per_byte number of samples per byte
     */
      virtual void pack (const dsp::TimeSeries * input,
                dsp::BitSeries * output,
                const int nbit,
                const int digi_min,
                const int digi_max,
                const float digi_mean,
                const float digi_scale,
                const float effective_scale,
                const int samp_per_byte
              ) = 0;

      /**
       * @brief Copy weights from input WeightedTimeSeries to output BitSeries.
       * Transposes the order from FPT WeightedTimeSeries to TFP BitSeries.
       *
       * @param weighted_input the input WeightedTimeSeries
       * @param output_weights the output BitSeries
       */
      virtual void copy_weights(const dsp::WeightedTimeSeries * weighted_input, dsp::BitSeries * output_weights) = 0;

      /**
       * @brief Perform internal setup operations for the engine.
       *
       * @param user pointer to the transformation instance that uses this engine
       */
      virtual void setup (GenericVoltageDigitizer* user);

      /**
       * @brief Set the scratch space to be used by the engine.
       *
       * @param scratch_space pointer to memory usable as scratch space
       * @param scratch_space_size size of the scratch space in bytes
       */
      virtual void set_scratch_space (void * scratch_space, size_t scratch_space_size) = 0;

  };
} // namespace dsp

#endif // __dsp_GenericVoltageDigitizer_h
