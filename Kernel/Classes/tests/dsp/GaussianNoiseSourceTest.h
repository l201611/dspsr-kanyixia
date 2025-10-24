/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GaussianNoiseSource.h"
#include "dsp/TimeSeries.h"

#include <gtest/gtest.h>
#include <vector>
#include <tuple>

#ifndef __dsp_GaussianNoiseSourceTest_h
#define __dsp_GaussianNoiseSourceTest_h

namespace dsp::test
{

  /**
   * @brief Unit test class for testing the GaussianNoiseSource functionality.
   * This class is derived from `::testing::TestWithParam`, allowing parameterized tests
   * with different configurations of the GaussianNoiseSource. The parameters are passed
   * as a tuple containing:
   *
   * - `dsp::TimeSeries::Order` : The data ordering format to test.
   *
   */
  class GaussianNoiseSourceTest : public ::testing::TestWithParam<dsp::TimeSeries::Order>
  {
  public:

    /**
     * @brief Construct a new GaussianNoiseSourceTest object
     *
     */
    GaussianNoiseSourceTest();

    /**
     * @brief Destroy the GaussianNoiseSourceTest object
     *
     */
    ~GaussianNoiseSourceTest() = default;

    /**
     * @brief Construct and configure the dsp::test::GaussianNoiseSource object to be tested
     *
    */
    dsp::test::GaussianNoiseSource* source_under_test(int niterations = 1);

    /**
     * @brief helper function for performing GaussianNoiseSource transform.
     */
    bool prepare_source(std::shared_ptr<dsp::test::GaussianNoiseSource> source);

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

    //! Assert that the data are gaussian
    //! The threshold allows for setting of far the sample mean and variance can be from a standard gaussian
    //! 6.0 = 3.4 false positives per a million tests
    void assert_gaussian_noise(std::shared_ptr<GaussianNoiseSource> source, double threshold = 6.0);

  protected:
    void SetUp() override;

    void TearDown() override;
  };

} // namespace dsp::test

#endif // __dsp_GaussianNoiseSourceTest_h
