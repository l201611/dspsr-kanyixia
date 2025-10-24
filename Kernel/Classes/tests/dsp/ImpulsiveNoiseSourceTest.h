/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ImpulsiveNoiseSource.h"
#include "dsp/TimeSeries.h"

#include <gtest/gtest.h>
#include <vector>
#include <tuple>

#ifndef __dsp_ImpulsiveNoiseSourceTest_h
#define __dsp_ImpulsiveNoiseSourceTest_h

namespace dsp::test
{

  /**
   * @brief Unit test class for testing the ImpulsiveNoiseSource functionality.
   * This class is derived from `::testing::TestWithParam`, allowing parameterized tests
   * with different configurations of the ImpulsiveNoiseSource. The parameters are passed
   * as a tuple containing:
   *
   * - `dsp::TimeSeries::Order` : The data ordering format to test.
   *
   */
  class ImpulsiveNoiseSourceTest : public ::testing::TestWithParam<dsp::TimeSeries::Order>
  {
  public:

    /**
     * @brief Construct a new ImpulsiveNoiseSourceTest object
     *
     */
    ImpulsiveNoiseSourceTest();

    /**
     * @brief Destroy the ImpulsiveNoiseSourceTest object
     *
     */
    ~ImpulsiveNoiseSourceTest() = default;

    /**
     * @brief Construct and configure the dsp::test::ImpulsiveNoiseSource object to be tested
     *
    */
    dsp::test::ImpulsiveNoiseSource* source_under_test(int niterations = 1);

    /**
     * @brief helper function for performing ImpulsiveNoiseSource transform.
     */
    bool prepare_source(std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source);

    //! output container
    Reference::To<dsp::TimeSeries> output{nullptr};

    //! the ordering for the input timeseries
    dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;

    //! number of channels
    unsigned nchan{32};

    //! number of polarisations
    unsigned npol{2};

    //! number of dimensions
    unsigned ndim{2};

    //! number of time samples
    uint64_t ndat{64};

    //! Sampling rate, in seconds
    //! This is 1/32
    double tsamp_sec{0.03125};

    //! Period of impulses
    //! This means 2 impulses occur over the 64 samples
    double period{1.0};

    //! The duration of impulses
    //! given the tsamp & period, this is 2 samples in a row
    double impulse_duration{0.0625};

    //! Height of impulses
    float height{10.0};

    //! Phase offset of impulses
    float phase_offset{0.5};

    //! Assert that the generated data is correct. This delegates to correct ordering
    void assert_generated_data(std::shared_ptr<ImpulsiveNoiseSource> source, int iteration);

    //! Assert that the generated FPT data is correct
    void assert_fpt(std::shared_ptr<ImpulsiveNoiseSource> source, int iteration);

    //! Assert that the generated TFP data is correct
    void assert_tfp(std::shared_ptr<ImpulsiveNoiseSource> source, int iteration);

    //! Assert that the expected value for a given sample
    //! ichan, ipol, idat and idim are used for debugging when assertion fails
    void assert_expected_value(float actual_value, unsigned ichan, unsigned ipol, unsigned idat, unsigned idim, unsigned start_samp);

  protected:
    void SetUp() override;

    void TearDown() override;

  };

} // namespace dsp::test

#endif // __dsp_ImpulsiveNoiseSourceTest_h
