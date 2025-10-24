/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ResponseProduct.h"
#include <gtest/gtest.h>

#ifndef __dsp_ResponseProductTest_h
#define __dsp_ResponseProductTest_h

namespace dsp::test
{

/**
 * @brief A test fixture for the ResponseProduct class
 *
 */
class ResponseProductTest : public ::testing::Test
{
  public:

    //! Default constructor sets up some defaults
    ResponseProductTest();

    /**
     * @brief Construct and configure the dsp::ResponseProduct object to be tested
     *
    */
    dsp::ResponseProduct* new_device_under_test();

  protected:
};

} // namespace dsp::test

#endif // __dsp_ResponseProductTest_h
