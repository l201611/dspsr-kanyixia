/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos, Will Gauvin and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <gtest/gtest.h>
#include "dsp/ChanPolSelect.h"
#include "dsp/TimeSeries.h"
#include "dsp/WeightedTimeSeries.h"

#include <random>

#ifndef __dsp_ChanPolSelectTest_h
#define __dsp_ChanPolSelectTest_h

namespace dsp::test {

/**
 * @brief A value-parameterized test suite for ChanPolSelectTest class
 *
 * The test suite is parameterized by a std::tuple with two elements:
 * @param bool on_gpu true when the GPU implementation should be executed / tested
 * @param dsp::TimeSeries::Order the order of values in TimeSeries memory (TFP or FPT)
 * @param bool use_wts true when input and output containers are WeightedTimeSeries
 */
class ChanPolSelectTest : public ::testing::TestWithParam<std::tuple<bool,dsp::TimeSeries::Order,bool>>
{
  public:

    /**
     * @brief Construct a new ChanPolSelectTest object
     *
     */
    ChanPolSelectTest();

    /**
     * @brief Destroy the ChanPolSelectTest object
     *
     */
    ~ChanPolSelectTest() = default;

    /**
     * @brief Construct and configure a new dsp::ChanPolSelect object
     *
     * @return dsp::ChanPolSelect* pointer to newly constructed and configured dsp::ChanPolSelect object
     */
    dsp::ChanPolSelect* new_device_under_test();

    /**
     * @brief Helper function that initialises input and output containers
     * initialisation is dependent on the flag use_wts which determines wether
     * to use either dsp::TimeSeries or WeightedTimeSeries
    */
    void init_containers();

    /**
     * @brief Generate test data for input to ChanPolSelect::transform
     *
     */
    void generate_data();

    /**
     * @brief Compare data output by ChanPolSelect::transform against expectations
     *
     * @param nchan_weight_equals_nchan determines if nchan weight should equal nchan as test configuration
     * @param npol_weight_equals_npol determines if npol weight should equal npol as test configuration
     */
    void assert_data(bool nchan_weight_equals_nchan=false, bool npol_weight_equals_npol=false);

    /**
     * @brief Compare weights output by ChanPolSelect::transform against expectations
     *
     * @param nchan_weight_equals_nchan determines if nchan weight should equal nchan as test configuration
     * @param npol_weight_equals_npol determines if npol weight should equal npol as test configuration
     */
    void assert_weights(bool nchan_weight_equals_nchan=false, bool npol_weight_equals_npol=false);

    /**
     * @brief Helper function for asserting the channel and polarisation selection for the transformation.
     *
     * @param cps the transformation against which to check the selection configuration
     */
    void assert_transform_configurations(dsp::ChanPolSelect* cps);

    /**
     * @brief Helper function for asserting dsp::TimeSeries::OrderFPT ordered data
     *
     */
    void assert_fpt();

    /**
     * @brief Helper function for generating weights in WeightedTimeSeries
     *
     */
    void generate_wts();

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
     * @brief helper function for performing ChanPolSelect transform.
     * performs a function call of set_input, set_output, prepare, and operate
     *
     * @param cps transformation under test
     * @param print_exceptions flag that enables printing of exception stack traces to stdout
     * @return false if an error is encountered
     */
    bool perform_transform(dsp::ChanPolSelect* cps, bool print_exceptions);

    /**
     * @brief helper function for asserting the metrics of the ChanPolSelect transformation
     *
     * @param cps transformation under test
     */
    void assert_metrics(dsp::ChanPolSelect* cps);

    /**
     * @brief helper function for driving ChanPolSelect unit tests.
     *
     * @param random_start_index determines if ChanPolSelect uses a randomized offset value
     * @param nchan_weight_equals_nchan determines if nchan weight should equal nchan as test configuration
     * @param npol_weight_equals_npol determines if npol weight should equal npol as test configuration
     *
     */
    void test_driver(bool random_start_index=false, bool nchan_weight_equals_nchan=false, bool npol_weight_equals_npol=false);

    //! input container
    Reference::To<dsp::TimeSeries> input;

    //! output container
    Reference::To<dsp::TimeSeries> output;

    //! input wts container
    Reference::To<dsp::WeightedTimeSeries> input_wts;

    //! output wts container
    Reference::To<dsp::WeightedTimeSeries> output_wts;

    //! input device container
    Reference::To<dsp::TimeSeries> device_input;

    //! output device container
    Reference::To<dsp::TimeSeries> device_output;

    //! input wts device container
    Reference::To<dsp::WeightedTimeSeries> device_input_wts;

    //! output wts device container
    Reference::To<dsp::WeightedTimeSeries> device_output_wts;

    //! device memory manager
    Reference::To<dsp::Memory> device_memory;

    //! number of channels
    unsigned nchan{32};

    //! number of polarisations
    unsigned npol{2};

    //! number of dimensions
    unsigned ndim{2};

    //! number of time samples
    uint64_t ndat{32};

    //! number of time samples per weight in WTS
    uint64_t ndat_per_weight{16};

    //! the start channel index used in the test
    unsigned start_channel_index{0};

    //! the number of channels to select in the test
    unsigned number_of_channels_to_keep{0};

    //! the start polarization index used in the test
    unsigned start_polarization_index{0};

    //! the number of polarizations to keep in the test
    unsigned number_of_polarizations_to_keep{0};

    //! list of signal states that can be tested
    std::vector<Signal::State> states;

  protected:

    void SetUp() override;

    void TearDown() override;

    //! Set true when test should be performed on GPU
    bool on_gpu = false;

    //! Set true when test should WeightedTimeSeries input and output containers
    bool use_wts = false;

    //! Order of TimeSeries data
    dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;

  private:

    /**
     * @brief Generate the expected floating point data value for the configured channel, polarisation and sample
     *
     * @param ichan channel number
     * @param ipol polarisation number
     * @param idat sample number
     * @return float expected input value
     */
    float get_expected_dat(unsigned ichan, unsigned ipol, uint64_t idat);

    /**
     * @brief Get the expected weight value for the configured channel, polarisation and sample
     *
     * @param ichan channel number
     * @param ipol polarisation number
     * @param idat sample number
     * @return uint16_t expected weight value
     */
    uint16_t get_expected_weight(unsigned ichan, unsigned ipol, uint64_t iweight);

    /**
     * @brief Generate a random number within the range.
     *
     * @param min minimum allowed value (inclusive)
     * @param max maximum allowed value (inclusive)
     * @return int randomly generated value
     */
    int generate_random_number(int min, int max);

    //! random device to use in seeding the value generator
    std::random_device rd{};

    //! random number generator to use in the normal distribution generation
    std::mt19937 generator{rd()};
};

} // namespace dsp::test

#endif // __dsp_ChanPolSelectTest_h
