/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Convolution.h"
#include "dsp/TransformationTestHelper.h"

#include <gtest/gtest.h>

#ifndef __dsp_ConvolutionTest_h
#define __dsp_ConvolutionTest_h

namespace dsp::test
{
  /**
   * @brief a struct to keep track of all the parameters used with the tests.
   *
   * This is generated during setup of parameterised tests to allow different
   * combinations of: if the tests should be on the GPU or no, the Timeseries ordering,
   * whether or not to dump the scales and offsets, the number or polarisations, the
   * type of statistics calculator, the signal state, and the mode of data.
   */
  class ConvolutionTestParam
  {
  public:
    //! Whether the test should run on the GPU or not
    bool on_gpu = false;

    //! The signal state of the input signal (Nyquist or Analytic)
    Signal::State state = Signal::Analytic;

    //! Test matrix convolution
    bool matrix = false;

    std::string get_name();
  };

/**
 * @brief A test fixture for the Convolution class
 *
 */
class ConvolutionTest : 
  public ::testing::TestWithParam<ConvolutionTestParam>,
  public TransformationTestHelper<TimeSeries,TimeSeries>
{
  public:

    //! Construct the input and output TimeSeries and Response function
    ConvolutionTest();

    //! Construct and configure the dsp::Convolution object to be tested
    dsp::Convolution* new_device_under_test();

  protected:

    //! The frequency response function applied during the Convolution operation
    Reference::To<Response> response;

    //! Initialize the current test parameters
    void SetUp() override;

    //! Reset the current test parameters to their default
    void TearDown() override;

    //! The parameters of the current test
    ConvolutionTestParam param;
};

} // namespace dsp::test

#endif // __dsp_ConvolutionTest_h
