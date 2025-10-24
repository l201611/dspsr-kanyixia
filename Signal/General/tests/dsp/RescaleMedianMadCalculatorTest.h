/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <memory>

#include <gtest/gtest.h>
#include <dsp/RescaleMedianMadCalculator.h>
#include <dsp/TimeSeries.h>

#ifndef __dsp_RescaleMedianMadCalculatorTest_h
#define __dsp_RescaleMedianMadCalculatorTest_h

namespace dsp::test {

  /**
   * @brief tests the RescaleMedianMadCalculator functionality
   */
  class RescaleMedianMadCalculatorTest : public ::testing::TestWithParam<dsp::TimeSeries::Order>
  {
    public:

      /**
       * @brief Construct a new RescaleMedianMadCalculatorTest object
       *
       */
      RescaleMedianMadCalculatorTest();

      /**
       * @brief Destroy the RescaleMedianMadCalculatorTest object
       *
       */
      ~RescaleMedianMadCalculatorTest() = default;

      /**
       * @brief Generate test data for input to RescaleMedianMadCalculator
       *
       */
      void generate_data();

      /**
       * @brief Helper function for populating the input container
       * with a dsp::TimeSeries::OrderFPT ordered data with
       * determined offset and scales
       *
       */
      void generate_fpt();

      /**
       * @brief Helper function for populating the input container
       * with a dsp::TimeSeries::OrderTFP ordered data
       * determined offset and scales
       *
       */
      void generate_tfp();

      /**
       * @brief Compare data output by RescaleMedianMadCalculator against expectations
       *
       */
      void assert_data();

      /**
       * @brief Helper function for asserting dsp::TimeSeries::OrderFPT
       * ordered data has the correctly calculated scales and offsets
       */
      void assert_fpt();

      /**
       * @brief Helper function for asserting dsp::TimeSeries::OrderTFP
       * ordered data has the correctly calculated scales and offsets
       */
      void assert_tfp();

      /**
       * @brief Assert the that the expected scales and offsets were calculated.
       *
       * This method performs a sort on the sample data and then takes the element at index (ndat - 1)/2
       * which accounts for 0-offset indexing (e.g. if ndat = 5 then the index would be 2 or the 3rd element).
       *
       * It then does an in-place fabs(sample - median) and then resorts the data and gets the value at the above index.
       *
       * This is technically different to the median-of-medians algorithm used in the RescaleMedianMadCalculator which
       * uses a recursive pivoting and it doesn't completely sort the data, and the absolute deviation is not calculated
       * in-place but using temporary scratch space.
       *
       * @param data the timeseries data converted to a multi-dimensional array of floats with order of [ipol][ichan][idat][idim]
       */
      void assert_expected_statistics(std::vector<std::vector<std::vector<float>>> data);

      /**
       * @brief helper function for performing RescaleMedianMadCalculator computation.
       *
       * This helper function performs init, sample_data and then compute on the calculator
       * when the total number of samples and the number nsample in sample_data are the same
       * (i.e. only need to call sample_data once).
       *
       * @return false if an error is encountered
       *
       */
      bool perform_computation();

      //! The calculator under test
      std::shared_ptr<dsp::RescaleMedianMadCalculator> calculator{nullptr};

      //! input timeseries
      Reference::To<dsp::TimeSeries> input;

      //! number of channels
      unsigned nchan{2};

      //! number of polarisations
      unsigned npol{2};

      //! number of dimensions
      unsigned ndim{2};

      //! number of time samples
      uint64_t ndat{32};

      //! scale factor to apply to convert MAD to an standard deviation estimate
      float scale_factor = dsp::RescaleMedianMadCalculator::DEFAULT_SCALE_FACTOR;

      //! Offset for input data
      float input_offset{0};

      //! Scale/STD for input data
      float input_scale{10.0};

    protected:

      //! Set up the tests
      void SetUp() override;

      //! Tear down the test
      void TearDown() override;

      //! Order of TimeSeries data
      dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;
  };

} // namespace dsp::test

#endif // __dsp_RescaleMedianMadCalculatorTest_h
