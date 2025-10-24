/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SumSource.h"
#include "dsp/GaussianNoiseSource.h"
#include "dsp/ImpulsiveNoiseSource.h"
#include "dsp/SumSource.h"
#include "dsp/TimeSeries.h"

#include <gtest/gtest.h>
#include <vector>
#include <tuple>

#ifndef __dsp_SumSourceTest_h
#define __dsp_SumSourceTest_h

namespace dsp::test
{

  /**
   * @brief Unit test class for testing the SumSource functionality.
   * This class is derived from `::testing::TestWithParam`, allowing parameterized tests
   * with different configurations of the SumSource. The parameters are passed
   * as a tuple containing:
   *
   * - `dsp::TimeSeries::Order` : The data ordering format to test.
   *
   */
  class SumSourceTest : public ::testing::TestWithParam<dsp::TimeSeries::Order>
  {
  public:

    /**
     * @brief Construct a new SumSourceTest object
     *
     */
    SumSourceTest();

    /**
     * @brief Destroy the SumSourceTest object
     *
     */
    ~SumSourceTest() = default;

    /**
     * @brief Construct and configure the dsp::test::SumSource object to be tested
     *
    */
    dsp::test::SumSource* source_under_test(int niterations = 1);

    /**
     * @brief helper function for performing SumSource transform.
     */
    bool prepare_source(std::shared_ptr<dsp::test::SumSource> source);

    //! output container
    Reference::To<dsp::TimeSeries> output{nullptr};

    //! gaussian noise source
    Reference::To<dsp::test::GaussianNoiseSource> gaussian_noise{nullptr};

    //! impulsive noise source
    Reference::To<dsp::test::ImpulsiveNoiseSource> impulsive_noise{nullptr};

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

    //! Use to assert the output data is the sum of the outputs of the individual sources
    void assert_data(std::shared_ptr<SumSource> source, bool all_zeros = false);

    //! Use to assert the FPT ordered output data is the sum of the outputs of the individual sources
    void assert_fpt(std::shared_ptr<SumSource> source, bool all_zeros = false);

    //! Use to assert the TFP ordered output data is the sum of the outputs of the individual sources
    void assert_tfp(std::shared_ptr<SumSource> source, bool all_zeros = false);

  protected:
    void SetUp() override;

    void TearDown() override;
  };

} // namespace dsp::test

#endif // __dsp_SumSourceTest_h
