/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/ImpulsiveNoiseSourceTest.h"

#include "dsp/GtestMain.h"
#include "dsp/TimeSeries.h"
#include "true_math.h"



#include <algorithm>
#include <random>
#include <cmath>
#include <array>
#include <chrono>
#include <string>
#include <cstdlib>

//! main method passed to googletest
int main(int argc, char *argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test
{

  ImpulsiveNoiseSourceTest::ImpulsiveNoiseSourceTest()
  {
    srand (static_cast <unsigned> (time(nullptr)));
  }

  void ImpulsiveNoiseSourceTest::SetUp()
  {
    if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
    {
      order = GetParam();
    }

    output = new dsp::TimeSeries;
    output->set_order(order);
  }

  void ImpulsiveNoiseSourceTest::TearDown()
  {
    output = nullptr;
  }

  dsp::test::ImpulsiveNoiseSource* ImpulsiveNoiseSourceTest::source_under_test(int niterations)
  {
    Reference::To<dsp::test::ImpulsiveNoiseSource> source = new dsp::test::ImpulsiveNoiseSource(niterations);
    source->set_period(period);
    source->set_impulse_duration(impulse_duration);
    source->set_height(height);
    source->set_phase_offset(phase_offset);
    source->set_output_order(order);
    return source.release();
  }

  bool ImpulsiveNoiseSourceTest::prepare_source(std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source)
  try
  {
    output->set_nchan(nchan);
    output->set_npol(npol);
    output->set_ndim(ndim);
    output->set_rate(tsamp_sec);
    output->resize(ndat);

    source->set_output(output);
    source->prepare();
    return true;
  }
  catch (std::exception &exc)
  {
    std::cerr << "Exception Caught: " << exc.what() << std::endl;
    return false;
  }
  catch (Error &error)
  {
    std::cerr << "Error Caught: " << error << std::endl;
    return false;
  }

  void ImpulsiveNoiseSourceTest::assert_expected_value(float actual_value, unsigned ichan, unsigned ipol, unsigned idat, unsigned idim, unsigned start_samp)
  {
    const auto isamp = start_samp + idat;

    const double duty_cycle = impulse_duration / period;
    const double phase_per_sample = tsamp_sec / period;
    const double frac_phase = fmod(static_cast<double>(isamp) * phase_per_sample + static_cast<double>(phase_offset), 1.0);

    float expected_value = frac_phase < duty_cycle ? height : 0.0;

    ASSERT_FLOAT_EQ(actual_value, expected_value) << "isamp=" << isamp
      << ", ichan=" << ichan
      << ", ipol=" << ipol
      << ", idat=" << idat
      << ", idim=" << idim
      << ", frac_phase=" << frac_phase
      << ", period=" << period
      << ", duty_cycle" << duty_cycle
      << ", start_samp=" << start_samp;
  }

  void ImpulsiveNoiseSourceTest::assert_generated_data(std::shared_ptr<ImpulsiveNoiseSource> source, int iteration)
  {
    switch (source->get_output()->get_order())
    {
    case dsp::TimeSeries::OrderFPT:
      assert_fpt(source, iteration);
      break;
    case dsp::TimeSeries::OrderTFP:
    default:
      assert_tfp(source, iteration);
      break;
    }
  }

  void ImpulsiveNoiseSourceTest::assert_fpt(std::shared_ptr<ImpulsiveNoiseSource> source, int iteration)
  {
    unsigned start_samp = static_cast<unsigned>(iteration * ndat);

    for (auto ichan = 0; ichan < nchan; ichan++)
    {
      for (auto ipol = 0; ipol < npol; ipol++)
      {
        float *ptr = source->get_output()->get_datptr(ichan, ipol);
        uint64_t ival = 0;
        for (auto idat = 0; idat < ndat; idat++)
        {
          for (auto idim = 0; idim < ndim; idim++, ival++)
          {
            assert_expected_value(ptr[ival], ichan, ipol, idat, idim, start_samp);
          }
        }
      }
    }
  }

  void ImpulsiveNoiseSourceTest::assert_tfp(std::shared_ptr<ImpulsiveNoiseSource> source, int iteration)
  {
    unsigned start_samp = static_cast<unsigned>(iteration * ndat);
    float *ptr = source->get_output()->get_dattfp();
    uint64_t ival = 0;

    for (auto idat = 0; idat < ndat; idat++)
    {
      for (auto ichan = 0; ichan < nchan; ichan++)
      {
        for (auto ipol = 0; ipol < npol; ipol++)
        {
          for (auto idim = 0; idim < ndim; idim++, ival++)
          {
            assert_expected_value(ptr[ival], ichan, ipol, idat, idim, start_samp);
          }
        }
      }
    }
  }

  TEST_P(ImpulsiveNoiseSourceTest, test_happy_path) // NOLINT
  {
    std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source(source_under_test());
    ASSERT_NE(source, nullptr);

    ASSERT_TRUE(prepare_source(source));
    source->set_period(1.0);
    source->set_impulse_duration(0.0625);

    ASSERT_FALSE(source->end_of_data());
    source->operate();
    ASSERT_TRUE(source->end_of_data());

    assert_generated_data(source, 0);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_P(ImpulsiveNoiseSourceTest, test_multiple_iterations) // NOLINT
  {
    std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source(source_under_test(2));
    ASSERT_NE(source, nullptr);

    ASSERT_TRUE(prepare_source(source));

    ASSERT_FALSE(source->end_of_data());
    source->operate();
    ASSERT_FALSE(source->end_of_data());
    assert_generated_data(source, 0);

    source->operate();
    ASSERT_TRUE(source->end_of_data());
    assert_generated_data(source, 1);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_P(ImpulsiveNoiseSourceTest, test_phase_offset) // NOLINT
  {
    phase_offset = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source(source_under_test());
    ASSERT_NE(source, nullptr);

    ASSERT_TRUE(prepare_source(source));
    ASSERT_EQ(source->get_phase_offset(), phase_offset);

    ASSERT_FALSE(source->end_of_data());
    source->operate();
    ASSERT_TRUE(source->end_of_data());
    assert_generated_data(source, 0);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_P(ImpulsiveNoiseSourceTest, test_height) // NOLINT
  {
    height = 100.0 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + 50.0;

    std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source(source_under_test());
    ASSERT_NE(source, nullptr);

    ASSERT_TRUE(prepare_source(source));
    ASSERT_EQ(source->get_height(), height);

    ASSERT_FALSE(source->end_of_data());
    source->operate();
    ASSERT_TRUE(source->end_of_data());
    assert_generated_data(source, 0);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_P(ImpulsiveNoiseSourceTest, test_period) // NOLINT
  {
    // have period a random number between 0.9 and 1.1
    period = 0.2 * static_cast <double> (rand()) / static_cast <double> (RAND_MAX) + 0.9;

    std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source(source_under_test());
    ASSERT_NE(source, nullptr);

    ASSERT_TRUE(prepare_source(source));
    ASSERT_EQ(source->get_period(), period);

    ASSERT_FALSE(source->end_of_data());
    source->operate();

    ASSERT_TRUE(source->end_of_data());
    assert_generated_data(source, 0);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_F(ImpulsiveNoiseSourceTest, test_invalid_period) // NOLINT
  {
    Reference::To<dsp::test::ImpulsiveNoiseSource> source = new dsp::test::ImpulsiveNoiseSource;

    ASSERT_ANY_THROW(source->set_period(0.0));
    ASSERT_ANY_THROW(source->set_period(-1.0));

    source = nullptr;
  }

  TEST_P(ImpulsiveNoiseSourceTest, test_impulse_duration) // NOLINT
  {
    impulse_duration = 0.02 * static_cast <double> (rand()) / static_cast <double> (RAND_MAX) + 0.09;

    std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source(source_under_test());
    ASSERT_NE(source, nullptr);

    ASSERT_TRUE(prepare_source(source));
    ASSERT_EQ(source->get_impulse_duration(), impulse_duration);

    ASSERT_FALSE(source->end_of_data());
    source->operate();

    ASSERT_TRUE(source->end_of_data());
    assert_generated_data(source, 0);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_F(ImpulsiveNoiseSourceTest, test_invalid_impulse_duration) // NOLINT
  {
    Reference::To<dsp::test::ImpulsiveNoiseSource> source = new dsp::test::ImpulsiveNoiseSource;

    ASSERT_NO_THROW(source->set_impulse_duration(0.0));
    ASSERT_ANY_THROW(source->set_impulse_duration(-1.0));

    source = nullptr;
  }

  TEST_P(ImpulsiveNoiseSourceTest, test_example_tsamp) // NOLINT
  {
    unsigned niterations = 3;
    tsamp_sec = 207.36 * 1e-6;
    ndat = 10000;
    nchan = 432;
    period = 0.2 * static_cast <double> (rand()) / static_cast <double> (RAND_MAX) + 0.9;
    impulse_duration = 0.02 * static_cast <double> (rand()) / static_cast <double> (RAND_MAX) + 0.09;
    height = 100.0 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + 50.0;

    std::shared_ptr<dsp::test::ImpulsiveNoiseSource> source(source_under_test(5));
    ASSERT_NE(source, nullptr);

    ASSERT_TRUE(prepare_source(source));
    ASSERT_EQ(source->get_output()->get_ndat(), ndat);
    ASSERT_EQ(source->get_output()->get_rate(), tsamp_sec);
    ASSERT_EQ(source->get_impulse_duration(), impulse_duration);
    ASSERT_EQ(source->get_period(), period);
    ASSERT_EQ(source->get_height(), height);

    ASSERT_FALSE(source->end_of_data());
    for (auto iteration = 0; iteration < niterations; iteration++)
    {
      source->operate();
      assert_generated_data(source, iteration);
    }

    source = nullptr;
  }

  INSTANTIATE_TEST_SUITE_P(
      ImpulsiveNoiseSourceTestSuite, ImpulsiveNoiseSourceTest,
      testing::Values(dsp::TimeSeries::OrderTFP, dsp::TimeSeries::OrderFPT),
      [](const testing::TestParamInfo<ImpulsiveNoiseSourceTest::ParamType> &info)
      {
        auto order = info.param;

        std::string name;

        if (order == dsp::TimeSeries::OrderFPT)
          name += "fpt";
        else
          name += "tfp";

        return name;
      }); // NOLINT

} // namespace dsp::test
