/***************************************************************************
 *
 *   Copyright (C) 2025 by Andrew Jameson and Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <gtest/gtest.h>
#include <dsp/TScrunch.h>
#include <dsp/TimeSeries.h>

#ifndef __dsp_TScrunchTest_h
#define __dsp_TScrunchTest_h

namespace dsp::test {

/**
 * @brief A value-parameterized test suite for TScrunchTest class
 *
 * The test suite is parameterized by a std::tuple with two elements:
 * @param bool on_gpu true when the GPU implementation should be executed / tested
 * @param dsp::TimeSeries::Order the order of values in TimeSeries memory (TFP or FPT)
 */
class TScrunchTest : public ::testing::TestWithParam<std::tuple<bool,dsp::TimeSeries::Order>>
{
  public:

    /**
     * @brief Construct a new TScrunchTest object
     *
     */
    TScrunchTest();

    /**
     * @brief Destroy the TScrunchTest object
     *
     */
    ~TScrunchTest() = default;

    /**
     * @brief Construct and configure the dsp::TScrunch object to be tested
     *
    */
    dsp::TScrunch* new_device_under_test();

    /**
     * @brief Generate test data for input to TScrunch::transform
     *
     */
    void generate_data();

    /**
     * @brief Compare data output by TScrunch::transform against expectations
     *
     */
    void assert_data();

    /**
     * @brief Get the expected input or output value for the timeseries.
     *
     * input data are computed as:
     *   ((ichan/nchan) * sample_index) + ipol
     * output daa are computed as the sum of corresponding input data for the scrunch factor
     *
     * @param ichan channel number
     * @param nchan total number of channels
     * @param ipol polarisation number
     * @param odat output time sample
     * @param sfactor time scrunch factor
     * @return float value that corresponds to the input
     */
    float get_expected_value(unsigned ichan, unsigned nchan, unsigned ipol, uint64_t odat, unsigned sfactor);

    /**
     * @brief Helper function for asserting transform configurations
     *
     */
    void assert_transform_configurations(std::shared_ptr<dsp::TScrunch> cps);

    /**
     * @brief Helper function for asserting dsp::TimeSeries::OrderFPT ordered data
     *
     * @param cps TScrunch pointer
     */
    void assert_fpt();

    /**
     * @brief Helper function for populating the input container
     * with a dsp::TimeSeries::OrderFPT ordered data
     *
     */
    void generate_fpt();

    /**
     * @brief Helper function for asserting dsp::TimeSeries::OrderTFP ordered data
     *
     */
    void assert_tfp();

    /**
     * @brief Helper function for populating the input container
     * with a dsp::TimeSeries::OrderTFP ordered data
     *
     */
    void generate_tfp();

    /**
     * @brief helper function for performing TScrunch transform.
     * performs a function call of set_input, set_output, prepare, and operate
     *
     * @return false if an error is encountered
     *
     */
    bool perform_transform(std::shared_ptr<dsp::TScrunch> cps);

    //! input container
    Reference::To<dsp::TimeSeries> input;

    //! output container
    Reference::To<dsp::TimeSeries> output;

    //! input to device container
    Reference::To<dsp::TimeSeries> device_input;

    //! output of device container
    Reference::To<dsp::TimeSeries> device_output;

    //! device memory manager
    Reference::To<dsp::Memory> device_memory;

    //! range of valid number of blocks/ to test
    const std::pair<unsigned, unsigned> nblocks_range = {2, 4};

    //! range of valid nchan to test
    const std::pair<unsigned, unsigned> nchan_range = {1, 31};

    //! range of valid ndat to test
    const std::pair<unsigned, unsigned> ndat_range = {2, 1023};

    //! range of valid nchan to test for small nchanpoldim tests
    const std::pair<unsigned, unsigned> small_nchan_range = {1, 4};

    //! range of valid nchan to test for large nchanpoldim tests
    const std::pair<unsigned, unsigned> large_nchan_range = {32, 128};

    //! range of valid nchan to test for very large nchanpoldim tests
    const std::pair<unsigned, unsigned> very_large_nchan_range = {32768, 65536};

    //! number of input time samples
    uint64_t ndat{0};

    //! sample number used to track time through multiple blocks
    uint64_t sample_number{0};

    //! number of time samples to scrunch in time
    unsigned tscrunch_factor{0};

    //! list of signal states that can be tested
    std::vector<Signal::State> states;

  protected:

    void SetUp() override;

    void TearDown() override;

    /**
     * @brief Generate a random number within a range.
     *
     * @param min minimum value of the range
     * @param max maximum value of the range
     * @return unsigned random value within the range
     */
    unsigned generateRandomNumber(unsigned min, unsigned max);

    //! Set true when test should be performed on GPU
    bool on_gpu = false;

    //! Order of TimeSeries data
    dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;
};

} // namespace dsp::test

#endif // __dsp_TScrunchTest_h
