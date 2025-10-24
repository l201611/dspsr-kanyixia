/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbank.h"
#include "dsp/TransformationTestHelper.h"
#include "dsp/Source.h"

#include <gtest/gtest.h>

#ifndef __dsp_InverseFilterbankTest_h
#define __dsp_InverseFilterbankTest_h

namespace dsp::test
{
  /**
   * @brief Manages unit test configuration parameters
   *
   */
  class InverseFilterbankTestParam
  {
  public:

    //! Whether the test should run on the GPU or not
    bool on_gpu = false;

    //! Returns a unique string that describes the test parameters
    std::string get_name();
  };

/**
 * @brief A test fixture for the InverseFilterbank class
 *
 */
class InverseFilterbankTest : 
  public ::testing::TestWithParam<InverseFilterbankTestParam>,
  public TransformationTestHelper<TimeSeries,TimeSeries>
{
  public:

    //! Construct and configure the dsp::InverseFilterbank object to be tested
    dsp::InverseFilterbank* new_device_under_test();

    //! Load input data from a file in $DSPSR_TEST_DATA/inverse_filterbank
    void load_input_from_file (const std::string& filename);

    //! Perform temporal_fidelity test
    /*!
      @param filename name of file containing input test vector
      @param input_nchan expected number of input channels
      @param impulse_idat expected index of delta function after inversion
    */
    void test_temporal_fidelity (const std::string& filename, unsigned input_nchan, unsigned impulse_idat);

  protected:

    //! Initialize the current test parameters
    void SetUp() override;

    //! Reset the current test parameters to their default
    void TearDown() override;

    //! The parameters of the current test
    InverseFilterbankTestParam param;

    //! The source of test data loaded from files
    Reference::To<dsp::Source> source;
};

} // namespace dsp::test

#endif // __dsp_InverseFilterbankTest_h
