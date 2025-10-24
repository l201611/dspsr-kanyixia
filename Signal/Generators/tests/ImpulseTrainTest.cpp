/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ImpulseTrainTest.h"
#include "dsp/GtestMain.h"

using namespace std;

//! main method passed to googletest
int main(int argc, char *argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test
{
  TEST_F(ImpulseTrainTest, test_construct_delete) // NOLINT
  {
    Reference::To<dsp::ImpulseTrain> generator = new dsp::ImpulseTrain;
    ASSERT_NE(generator, nullptr);
    generator = nullptr;
    ASSERT_EQ(generator, nullptr);
  }

  TEST_F(ImpulseTrainTest, test_operation) try // NOLINT
  {
    Reference::To<dsp::ImpulseTrain> generator = new dsp::ImpulseTrain;
    Reference::To<dsp::TimeSeries> signal = new dsp::TimeSeries;

    generator->set_output(signal);

    if (Operation::verbose)
      cerr << "ImpulseTrainTest test_operation operate without period set" << endl;

    // the period has not been set
    EXPECT_THROW(generator->operate(), ::Error);

    constexpr unsigned period = 5;
    generator->set_period_samples(period);

    if (Operation::verbose)
      cerr << "ImpulseTrainTest test_operation operate without block_size set" << endl;

    generator->operate();

    // the block size has not been set
    EXPECT_EQ(signal->get_ndat(), 0);

    constexpr unsigned block_size = 32;
    generator->set_block_size(block_size);

    if (Operation::verbose)
      cerr << "ImpulseTrainTest test_operation operate with block_size set" << endl;

    EXPECT_EQ(generator->get_block_size(), block_size);

    generator->operate();

    if (Operation::verbose)
      cerr << "ImpulseTrainTest test_operation test that block size is transferred" << endl;

    EXPECT_EQ(signal->get_ndat(), block_size);

    // For now, ImpulseTrain outputs a single channel, single polarization time series
    EXPECT_EQ(signal->get_nchan(), 1);
    EXPECT_EQ(signal->get_npol(), 1);

    auto data = signal->get_datptr();

    unsigned ndim = signal->get_ndim();
    unsigned nfloat = block_size * ndim;
    unsigned period_float = period * ndim;

    if (Operation::verbose)
      cerr << "ImpulseTrainTest test_operation test data after first call to operate" << endl;

    unsigned errors = 0;
    for (unsigned idat=0; idat < nfloat; idat++)
    {
      if (idat % period_float == 0 && data[idat] != 1.0)
        errors ++;

      if (idat % period_float != 0 && data[idat] != 0.0)
        errors ++;
    }

    EXPECT_EQ(errors, 0);

    generator->operate();

    if (Operation::verbose)
      cerr << "ImpulseTrainTest test_operation test data after second call to operate" << endl;

    for (unsigned idat=0; idat < nfloat; idat++)
    {
      unsigned jdat = nfloat + idat;
      if (jdat % period_float == 0 && data[idat] != 1.0)
        errors ++;

      if (jdat % period_float != 0 && data[idat] != 0.0)
        errors ++;
    }

    if (errors && Operation::verbose)
    {
      for (unsigned idat=0; idat < nfloat; idat++)
        cerr << data[idat] << " ";
      cerr << endl;
    }

    EXPECT_EQ(errors, 0);

  }
  catch (Error& error)
  {
    cerr << "ImpulseTrainTest test_operation " << error << endl;
  }
}
