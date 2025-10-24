/***************************************************************************
 *
 *   Copyright (C) 2022 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Signal_Generators_tests_ImpulseTrainTest_h
#define __dsp_Signal_Generators_tests_ImpulseTrainTest_h

#include "dsp/ImpulseTrain.h"
#include <gtest/gtest.h>

namespace dsp::test 
{
  class ImpulseTrainTest : public ::testing::TestWithParam<const char *>
  {
  };

} // namespace dsp::test

#endif // !define(__dsp_Signal_Generators_tests_ImpulseTrainTest_h)
