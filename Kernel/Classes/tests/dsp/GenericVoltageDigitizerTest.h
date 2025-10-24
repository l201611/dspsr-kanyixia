/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericVoltageDigitizer.h"
#include "dsp/TimeSeries.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/BitSeries.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <vector>
#include <tuple>

#ifndef __dsp_GenericVoltageDigitizerTest_h
#define __dsp_GenericVoltageDigitizerTest_h

namespace dsp::test
{

  /**
   * @brief Unit test class for testing the GenericVoltageDigitizer functionality.
   * This class is derived from `::testing::TestWithParam`, allowing parameterized tests
   * with different configurations of the GenericVoltageDigitizer. The parameters are passed
   * as a tuple containing:
   *
   * @param int : The number of bits per sample to test.
   * @param dsp::TimeSeries::Order : The data ordering format to test.
   * @param bool : A flag indicating whether to test with a CUDA capable gpu when available.
   * @param bool : use_wts true when input and output containers are WeightedTimeSeries
   *
   */
  class GenericVoltageDigitizerTest : public ::testing::TestWithParam<std::tuple<int, dsp::TimeSeries::Order, bool, bool>>
  {
  public:

    /**
     * @brief Construct a new GenericVoltageDigitizerTest object
     *
     */
    GenericVoltageDigitizerTest();

    /**
     * @brief Destroy the GenericVoltageDigitizerTest object
     *
     */
    ~GenericVoltageDigitizerTest() = default;

    /**
     * @brief Construct and configure the dsp::GenericVoltageDigitizer object to be tested
     *
    */
    dsp::GenericVoltageDigitizer* new_device_under_test();

    /**
     * @brief Helper function for asserting output of generated data given for the current timeseries ordering
     */
    void assert_generated_data();

    /**
     * @brief Helper function for asserting the output data based on current timeseries ordering
     * from known fixed data
     */
    void assert_known_data();

    /**
     * @brief Helper function for asserting dsp::TimeSeries::OrderTFP ordered data
     * from known fixed data
     */
    void assert_tfp_known_data();

    /**
     * @brief Helper function for asserting dsp::TimeSeries::OrderFPT ordered data
     * from known fixed data
     */
    void assert_fpt_known_data();

    /**
     * @brief Helper function for asserting output Weights BitSeries container
     */
    void assert_output_wts();

    /**
     * @brief unpacks the value when NBIT<=8
     *
     * For NBIT=1 this just returns the bit as an integer (i.e. 0 or 1)
     * For NBIT=8 this would return the value as a signed integer
     *
     * For other NBIT this uses the outidx and NBIT to determine which bits represent the
     * current value. If the most significant bit, of the nbits, is set this method will
     * return the bits as a negative number.
     *
     * @param value the current input byte
     * @param outidx the current output index based off the current idat, ichan, ipol and idim
     *
     * @returns the signed integer value based on the which bits represent the outidx
     */
    int unpack_value(unsigned char value, unsigned outidx);

    /**
     * @brief Helper function to calculate the expected value given the input value
     *
     * @param input_value the value from the input time series
     *
     * @return the expected integer value given the digitizer's configuration.
     */
    int calculate_expected_value(float input_value);

    /**
     * @brief helper function for preparing the input and output(s) for the
     * GenericVoltageDigitizer transform.
     *
     * Calls set_input, set_output and if \c use_wts is true, calls set_output_weights.
     * If on_gpu is true, ensures the containers are located on the gpu device.
     *
     * @param gvd A shared pointer to the GenericVoltageDigitizer instance.
     */
    void prepare_transform_input_and_outputs(std::shared_ptr<dsp::GenericVoltageDigitizer> gvd);

    /**
     * @brief helper function for performing GenericVoltageDigitizer transform.
     *
     * Uses prepare_transform_input_and_outputs to setup the inputs and outputs,
     * then calls prepare and operate.
     *
     * @param gvd A shared pointer to the GenericVoltageDigitizer instance.
     * @return false if an error is encountered
     */
    bool perform_transform(std::shared_ptr<dsp::GenericVoltageDigitizer> gvd);

    /**
     * @brief assert the performance metrics of the GenericVoltageDigitizer
     *
     * @param gvd A shared pointer to the GenericVoltageDigitizer instance.
     */
    void assert_metrics(std::shared_ptr<dsp::GenericVoltageDigitizer> gvd);

    uint16_t generate_known_fpt_weights(uint64_t nval);

    //! input container
    Reference::To<dsp::TimeSeries> input{nullptr};

    //! output container
    Reference::To<dsp::BitSeries> output{nullptr};

    //! input container for WeightedTimeSeries
    Reference::To<dsp::WeightedTimeSeries> input_wts;

    //! output container for weights BitSeries
    Reference::To<dsp::BitSeries> output_wts;

    //! input to device container
    Reference::To<dsp::TimeSeries> device_input;

    //! output of device container
    Reference::To<dsp::BitSeries> device_output;

    //! input to device container for WeightedTimeSeries
    Reference::To<dsp::WeightedTimeSeries> device_input_wts;

    //! output of device container for weights BitSeries
    Reference::To<dsp::BitSeries> device_output_wts;

    //! device memory manager
    Reference::To<dsp::Memory> device_memory;

    //! the ordering for the input timeseries
    dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;

    //! number of bits per sample
    int nbit = 8;

    //! Set true when test should be performed on GPU
    bool on_gpu = false;

    //! Set true when test should be performed on GPU
    bool use_wts = false;

    //! number of channels
    unsigned nchan{32};

    //! number of polarisations
    unsigned npol{2};

    //! number of dimensions
    unsigned ndim{2};

    //! number of time samples
    uint64_t ndat{32};

    //! number of time samples per weight in WTS
    uint64_t static const ndat_per_weight{16};

    //! list of signal states that can be tested
    std::vector<Signal::State> states;

  protected:
    void SetUp() override;

    void TearDown() override;

    //! pick normally distributed number with a scale and mean.
    void generate_random_data(float *data, size_t data_size, float scale = 1.0, float mean = 0.0);

    //! Helper function used when generating weights
    uint16_t expected_weight(unsigned ichan, unsigned ipol, uint64_t iweight);

     //! Helper function for generating weights in WeightedTimeSeries
    void generate_weights(int _ndat_per_weight=ndat_per_weight);

    //! setup zeroed data with alternative -0.0 and 0.0 values
    void setup_zeroed_data();

    //! setup random timeseries data given an input timeseries order
    void setup_random_timeseries_data();

    //! setup known data to the input Timeseries data given timeseries input order
    void setup_known_data_timeseries();

    //! setup known data to the input Timeseries based on nchan and npol
    void setup_nchanpol_timeseries();

    /**
     * @brief Encode the current time sample, channel, polarisation and dimension to a known floating point value.
     *
     * This method allows testing of digitisation to know we pack the value in the correct order given the
     * given time sample, channel, polarisation and dimension.
     *
     * @param idat the current time sample
     * @param ichan the current channel
     * @param ipol the current polarisation
     * @param idim the current dimension
     */
    float encode_idat_ichan_ipol_idim(uint64_t idat, unsigned ichan, unsigned ipol, unsigned idim);

    //! get the expected output data when input was TFP ordered timeseries
    const int8_t *expected_output_data_tfp();

    //! get the expected output data when input was FPT ordered timeseries
    const int8_t *expected_output_data_fpt();

#ifdef HAVE_CUDA
    //! @brief CUDA stream handle.
    cudaStream_t stream{nullptr};
#endif

  };

} // namespace dsp::test

#endif // __dsp_GenericVoltageDigitizerTest_h
