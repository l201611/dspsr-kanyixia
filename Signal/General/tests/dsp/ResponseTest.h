/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Response.h"
#include <gtest/gtest.h>

#ifndef __dsp_ResponseTest_h
#define __dsp_ResponseTest_h

namespace dsp::test
{

/**
 * @brief A test fixture for the Response class
 *
 */
class ResponseTest : public ::testing::Test
{
  public:

    ResponseTest();
    
    /**
     * @brief Construct and configure the dsp::Response object to be tested
     *
    */
    dsp::Response* new_device_under_test();

  protected:
};

} // namespace dsp::test

#endif // __dsp_ResponseTest_h
