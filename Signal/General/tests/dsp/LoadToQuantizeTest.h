/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Andrew Jameson and Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <gtest/gtest.h>
#include <dsp/LoadToQuantize.h>
#include <dsp/Source.h>
#include <dsp/TimeSeries.h>
#include <dsp/TestSource.h>
#include <dsp/RescaleScaleOffsetDumpTestHelper.h>

#ifndef __dsp_LoadToQuantizeTest_h
#define __dsp_LoadToQuantizeTest_h

namespace dsp::test {

  /**
   * @brief a struct for handling the different parameter combinations for the parameterised tests.
   */
  struct LoadToQuantizeTestParam {
    //! the number bits for the output of GenericVoltageDigitizer
    int output_nbit;

    //! the ordering of the input timeseries data
    dsp::TimeSeries::Order order;

    //! indicator of whether to run on the GPU or CPU
    bool on_gpu;

    //! indicator of whether to use a weighted timeseries or not
    bool use_wts;

    //! indicator of whether to use the median/MAD algorithm or the default mean/std algorithm
    bool use_median_mad;
  };

  /**
   * @brief Unit test class for testing LoadToQuantize functionality.
   * This class is derived from `::testing::TestWithParam`, allowing parameterized tests
   * with different configurations of a LoadToQuantize instance. The parameters are passed
   * as a tuple containing:
   *
   * @param int : The number of bits per sample to test.
   * @param dsp::TimeSeries::Order : The data ordering format to test.
   * @param bool : A flag indicating whether to test with a CUDA capable gpu when available.
   * @param bool : If true, then input and output containers are WeightedTimeSeries
   *
   */
class LoadToQuantizeTest : public ::testing::TestWithParam<LoadToQuantizeTestParam>
{
  public:

    /**
     * @brief Construct a new LoadToQuantizeTest object
     *
     */
    LoadToQuantizeTest();

    /**
     * @brief Destroy the LoadToQuantizeTest object
     *
     */
    ~LoadToQuantizeTest() = default;

    /**
     * @brief Assert that the data DADA file header contains the expected meta-data parameter values.
     *
     * @param filename path of the DADA file to test
     */
    void assert_data_file_header(const std::string& filename);

    /**
     * @brief Assert that the data DADA file is as expected: the header contains the expected meta-data
     * and the data have the expected statistics.
     *
     * @param filename path of the DADA file to test
     */
    void assert_data_file(const std::string& filename);

    /**
     * @brief Assert that the weights DADA file header contains the expected meta-data parameter values.
     *
     * @param filename path of the DADA file to test
     */
    void assert_weights_file_header(const std::string& filename);

    /**
     * @brief Assert that the weights DADA file is as expected
     *
     * @param filename path of the DADA file to test
     */
    void assert_weights_file(const std::string& filename);

    /**
     * @brief Helper function to populate input container with correct ordering of data.
     *
     * This method delegates to the appropriate generate_fpt or generate_tfp method
     * depending upon the current value of order.
     */
    void generate_data();

    /**
     * @brief Helper function for populating the input container
     * with dsp::TimeSeries::OrderFPT ordered data
     *
     */
    void generate_fpt();

    /**
     * @brief Helper function for populating the input container
     * with dsp::TimeSeries::OrderTFP ordered data
     *
     */
    void generate_tfp();

    /**
     * @brief Helper function for populating the weights of the input container
     * when using WeightedTimeSeries
     *
     */
    void generate_weights();

    /**
     * @brief Prepare the LoadToQuantize::Config object with the configuration
     * parameters determined from the class attributes
     *
     */
    void prepare_config();

    /**
     * @brief Prepares the input TimeSeries, configuring the dimensions and meta-data
     *
     */
    void prepare_input();

    /**
     * @brief Helper function deletes files produced by tests
     *
     */
    void remove_output_files();

    /**
     * @brief Prepare and execute the pipeline/engine with the specified source iterations
     */
    void execute_engine();

    double unpack_value(const unsigned char *value, uint64_t idx);

    //! Configuration object for the LoadToQuantize pipeline
    dsp::LoadToQuantize::Config config;

    //! input container
    Reference::To<dsp::TimeSeries> input{};

    //! Set true when test should be performed on GPU
    bool on_gpu = false;

    //! Set true when weighted TimeSeries should be used
    bool use_wts = false;

    //! Set true when using median/MAD as a calculator
    bool use_median_mad = false;

    //! Indicator of using constant verses updating per chunk of data
    bool rescale_constant = false;

    //! The rescale interval time. If set to > 0.0 then this value will be set on the Rescale operation
    double rescale_interval = 0.0;

    //! file descriptor
    int fd = -1;

    unsigned header_size{4096};

    //! number of input channels
    unsigned nchan{32};

    //! number of input polarisations
    unsigned npol{2};

    //! number of input dimensions
    unsigned ndim{2};

    //! number of input time samples
    uint64_t ndat{1024};

    //! number of input bits per sample
    unsigned nbit{32};

    //! number of input weight polarizations
    unsigned npol_weight{1};

    //! number of input weight channels
    unsigned nchan_weight{nchan};

    //! number of time samples per weight
    unsigned ndat_per_weight{16};

    //! number of bits per output sample
    unsigned output_nbit{8};

    //! start channel index
    unsigned start_channel_index{0};

    //! number of channels to keep
    unsigned number_of_channels_to_keep{0};

    //! start polarization index
    unsigned start_polarization_index{0};

    //! number of polarizations to keep
    unsigned number_of_polarizations_to_keep{0};

    //! filename for the data OutputDADAFile
    std::string output_filename;

    //! filename for the weights OutputDADAFile
    std::string weights_output_filename;

    //! filename for the scales and offsets file
    std::string scale_offset_output_filename;

    //! the ordering for the input timeseries
    dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;

    //! number of pipeline iterations to perform
    unsigned num_iterations{1};

    //! the generated data to be sent through the pipeline
    std::vector<float> data;

    //! the generated weights to be sent through the pipeline
    std::vector<uint16_t> weights;

    //! test helper to assert the scales and offsets were dumped to a file correctly
    Reference::To<RescaleScaleOffsetDumpTestHelper> scale_offset_file_helper;

  protected:

    //! setup test infrastructure
    void SetUp() override;

    //! tear down test infrastructure
    void TearDown() override;
};

} // namespace dsp::test

#endif // __dsp_LoadToQuantizeTest_h
