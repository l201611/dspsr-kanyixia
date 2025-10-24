/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CyclicFold.h"
#include "dsp/TransformationTestHelper.h"

#include <gtest/gtest.h>

#ifndef __dsp_CyclicFoldTest_h
#define __dsp_CyclicFoldTest_h

namespace dsp::test
{
  /**
   * @brief a struct to keep track of all the parameters used with the tests.
   *
   * This is generated during setup of parameterised tests to allow different
   * combinations of:
   * - on_gpu: operations should be performed on the GPU
   * - state: real-valued (Signal::Nyquist) or complex-valued (Signal::Analytic)
   */
  class CyclicFoldTestParam
  {
  public:
    //! Whether the test should run on the GPU or not
    bool on_gpu = false;

    //! The signal state of the input signal (Nyquist or Analytic)
    Signal::State state = Signal::Analytic;

    std::string get_name();
  };

/**
 * @brief A test fixture for the CyclicFold class
 *
 */
class CyclicFoldTest : 
  public ::testing::TestWithParam<CyclicFoldTestParam>,
  public TransformationTestHelper<TimeSeries,PhaseSeries>
{
  public:

    //! Construct and configure the dsp::CyclicFold object to be tested
    dsp::CyclicFold* new_device_under_test();

  protected:

    //! Initialize the current test parameters
    void SetUp() override;

    //! Reset the current test parameters to their default
    void TearDown() override;

    //! The parameters of the current test
    CyclicFoldTestParam param;
};

} // namespace dsp::test

#endif // __dsp_CyclicFoldTest_h
