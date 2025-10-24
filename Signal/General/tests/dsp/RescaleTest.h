/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos, Andrew Jameson and Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <memory>

#include <gtest/gtest.h>
#include <dsp/Rescale.h>
#include <dsp/RescaleScaleOffsetDump.h>
#include <dsp/RescaleScaleOffsetDumpTestHelper.h>
#include <dsp/TimeSeries.h>
#include <gmock/gmock.h>

#ifndef __dsp_RescaleTest_h
#define __dsp_RescaleTest_h

namespace dsp::test {

  /**
   * @brief a struct to keep track of all the parameters used with the tests.
   *
   * This is generated during setup of parameterised tests to allow different
   * combinations of: if the tests should be on the GPU or no, the Timeseries ordering,
   * whether or not to dump the scales and offsets, the number or polarisations, the
   * type of statistics calculator, the signal state, and the mode of data.
   */
  struct TestParam {
    //! Whether the test should run on the GPU or not
    bool on_gpu;

    //! The mode of data, which allows for all zeros, or changing the scale and offset based on the channel and/or pol
    std::string mode;

    //! the order of the timeseries data
    dsp::TimeSeries::Order order;

    //! The stats calculator to determine scales and offsets.
    std::string calculator;

    //! The signal state of the input signal, this will define which number of polarisations are valid
    Signal::State state;

    //! The number of polarisations of the input data.
    unsigned npol;

    //! Indicator of whether to output the scales and offsets and to verify that they are correct based on the calculator
    /** Note only used of CPU based tests at the moment. */
    bool dump_scale_offset;
  } test_param_t;

  /**
   * @brief a mock/test double to assert that the Rescale's callback was called.
   */
  class ScalesUpdatedMock : public Reference::Able {
    public:
      MOCK_METHOD(void, scales_updated, (dsp::Rescale::update_record));
  };

  class RescaleTest : public ::testing::TestWithParam<TestParam>
  {
  public:

    enum Mode
    {
      None,
      HalfWave,
      LinearRamp,
      AllZero,
    };

    /**
     * @brief Construct a new RescaleTest object
     *
     */
    RescaleTest();

    /**
     * @brief Destroy the RescaleTest object
     *
     */
    ~RescaleTest() = default;

    /**
     * @brief Construct and configure the dsp::Rescale object to be tested
     *
     */
    void new_transform_under_test();

    /**
     * @brief Generate test data for input to Rescale::transformation
     *
     */
    void generate_data();

    /**
     * @brief Compare data output by Rescale::transformation against expectations
     *
     */
    void assert_data();

    /**
     * @brief Helper function for populating the input container
     * with a dsp::TimeSeries::OrderFPT ordered data with
     * determined offset and scales
     *
     */
    void generate_fpt();

    /**
     * @brief Helper function for asserting dsp::TimeSeries::OrderFPT
     * ordered data have zero mean and unit variance
     */
    void assert_fpt();

    /**
     * @brief Helper function for populating the input container
     * with a dsp::TimeSeries::OrderTFP ordered data
     * determined offset and scales
     *
     */
    void generate_tfp();

    /**
     * @brief Helper function for asserting dsp::TimeSeries::OrderTFP
     * ordered data have zero mean and unit variance
     */
    void assert_tfp();

    /**
     * @brief Assert the that the rescaled mean and standard deviation, computable from the sums and sums_sq meet the expected values.
     *
     * @param nsample number of samples with which to assert the statistics
     * @param std::vector<std::vector<double>> sumof the input values for each [chan][pol]
     * @param std::vector<std::vector<double>> sum of the square of the input values for each [chan][pol]
     */
    void assert_expected_rescaled_statistics(uint64_t nsample, const std::vector<std::vector<double>> &sums, const std::vector<std::vector<double>> &sums_sq);

    /**
     * @brief Helper function for asserting measured offsets
     * and scales matched the input values.
     */
    void assert_offsets_scales();

    /**
     * @brief assert scales and offset file dumped
     */
    void assert_scale_offset_dump();

    /**
     * @brief helper function for performing Rescale transform.
     * performs a function call of set_input, set_output, prepare, and operate
     *
     * @return false if an error is encountered
     *
     */
    bool perform_transform();

    //! utility method testing tests about using median/mad calculator
    void test_using_median_mad();

    //! utility method testing tests about using mean/std calculation (on gpu or cpu)
    void test_using_mean_std();

    /**
     * @brief assert the performance metrics of the rescale transformation
     *
     * @param rescale the rescale object to display metrics for
     */
    void assert_metrics(std::shared_ptr<dsp::Rescale> rescale);

    /**
     * @brief get the number of expected calls to the scales_updated mock
     *
     * @param isample the starting time sample of the input timeseries
     */
    int expected_calls_of_scales_updated(uint64_t isample);

    //! pointer to the class under test
    std::shared_ptr<dsp::Rescale> rescale;

    //! input container
    Reference::To<dsp::TimeSeries> input;

    //! output container
    Reference::To<dsp::TimeSeries> output;

    //! input container
    Reference::To<dsp::TimeSeries> device_input;

    //! output container
    Reference::To<dsp::TimeSeries> device_output;

    //! device memory manager
    Reference::To<dsp::Memory> device_memory;

    //! rescale engine
    Reference::To<dsp::Rescale::Engine> engine;

    //! Calculator to perform the statistics
    Reference::To<dsp::Rescale::ScaleOffsetCalculator> calculator;

    //! the rescale scale and offset data dump
    Reference::To<dsp::RescaleScaleOffsetDump> scale_offset_dump;

    //! test helper to assert the scales and offsets were dumped to a file correctly
    Reference::To<RescaleScaleOffsetDumpTestHelper> scale_offset_file_helper;

    //! The name
    std::string scale_offset_dump_filepath;

    //! number of channels
    unsigned nchan{2};

    //! number of polarisations
    unsigned npol{2};

    //! number of dimensions
    unsigned ndim{1};

    //! state of the input signal
    Signal::State state;

    //! The name of the mode testing data
    std::string mode_name = "";

    //! The name of the calculator to use in parameterised tests
    std::string calculator_name = "MeanStd";

    //! Indicator of whether test is expecting scale and offset data to be dumped to file
    bool dump_scale_offset{false};

    //! number of time samples
    uint64_t ndat{16384};

    //! number of samples used in statistical calculations
    uint64_t nsample{0};

    //! flag for using exact samples, passed to Rescale::exact_samples during \ref perform_transform
    bool use_exact_samples{false};

    //! flag for constant thresholds, passed to Rescale::set_constant during \ref perform_transform
    bool use_constant_thresholds{false};

    class MockEngine : public Rescale::Engine
    {
    public:
      MockEngine()
      {
        ON_CALL(*this, set_calculator).WillByDefault([this](dsp::Rescale::ScaleOffsetCalculator* val){calculator = val;});
        ON_CALL(*this, get_calculator).WillByDefault([this](){return calculator;});
      }
      MOCK_METHOD(void, init, (const dsp::TimeSeries *input, uint64_t nsample, bool exact, bool constant_offset_scale), (override));
      MOCK_METHOD(void, transform, (const dsp::TimeSeries *input, dsp::TimeSeries *output), (override));
      MOCK_METHOD(const float*, get_offset, (unsigned ipol), (const, override));
      MOCK_METHOD(const float*, get_scale, (unsigned ipol), (const, override));
      MOCK_METHOD(const double*, get_freq_total, (unsigned ipol), (const, override));
      MOCK_METHOD(const double*, get_freq_squared_total, (unsigned ipol), (const, override));
      MOCK_METHOD(void, set_calculator, (Rescale::ScaleOffsetCalculator* _calculator), (override));
      MOCK_METHOD(const Rescale::ScaleOffsetCalculator*, get_calculator, (), (const, override));
      MOCK_METHOD(MJD, get_update_epoch, (), (const, override));
    protected:
      Rescale::ScaleOffsetCalculator* calculator = nullptr;
    };

    class MockCalculator : public Rescale::ScaleOffsetCalculator
    {
    public:
      MOCK_METHOD(void, init, (const dsp::TimeSeries* input, uint64_t ndat, bool output_time_total), (override));
      MOCK_METHOD(uint64_t, sample_data, (const dsp::TimeSeries* input, uint64_t start_dat, uint64_t end_dat, bool output_time_total), (override));
      MOCK_METHOD(void, compute, (), (override));
      MOCK_METHOD(void, reset_sample_data, (), (override));
      MOCK_METHOD(const double*, get_mean, (unsigned ipol), (const, override));
      MOCK_METHOD(const double*, get_variance, (unsigned ipol), (const, override));
    };

  protected:

    void SetUp() override;

    void TearDown() override;

    //! Set true when test should be performed on GPU
    bool on_gpu = false;

    //! Order of TimeSeries data
    dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;

    Mode input_mode = Mode::None;

    Mode offsets_scales_mode = Mode::None;

    //! return the Mode corresponding to the mode name
    Mode mode_name_to_mode(std::string mode_name);

    //! return the offset and scale for the mode, sample, pol and channel
    void get_offset_scale(Mode mode, uint64_t idat, unsigned ipol, unsigned ichan, float *offset, float *scale);
};

} // namespace dsp::test

#endif // __dsp_RescaleTest_h
