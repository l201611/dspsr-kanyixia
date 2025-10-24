/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_DADAHeaderTest_h
#define __dsp_DADAHeaderTest_h

#include "dsp/DADAHeader.h"
#include "Reference.h"
#include <gtest/gtest.h>

namespace dsp
{
  class ASCIIObservation;
  class DADAHeader;

namespace test 
{
  class DADAHeaderTest : public ::testing::TestWithParam<const char *>
  {
  public:
    /**
     * @brief Construct a new DADAHeaderTest object
     *
     */
    DADAHeaderTest();

    /**
     * @brief Destroy the DADAHeaderTest object
     *
     */
    ~DADAHeaderTest() = default;

    /**
     * @brief Verifies that DADA file header meta-data values match expected values.
     *
     */
    void assert_header(const char*, unsigned hdr_size = 0);

  protected:

    void SetUp() override;

    void TearDown() override;

    //! Used to fill the DADA header
    Reference::To<ASCIIObservation> observation;
  };

} // namespace test
} // namespace dsp

#endif // __dsp_DADAHeaderTest_h
