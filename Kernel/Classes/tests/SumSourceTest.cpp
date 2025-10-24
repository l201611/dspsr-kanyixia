/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/SumSourceTest.h"

#include "dsp/GtestMain.h"
#include "dsp/TimeSeries.h"
#include "true_math.h"



#include <algorithm>
#include <random>
#include <cmath>
#include <array>
#include <chrono>
#include <string>

//! main method passed to googletest
int main(int argc, char *argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test
{

  SumSourceTest::SumSourceTest()
  {
  }

  void SumSourceTest::SetUp()
  {
    if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
    {
      order = GetParam();
    }

    output = new dsp::TimeSeries;
    output->set_order(order);
  }

  void SumSourceTest::TearDown()
  {
    output = nullptr;
    gaussian_noise = nullptr;
    impulsive_noise = nullptr;
  }

  void SumSourceTest::assert_fpt(std::shared_ptr<SumSource> source, bool all_zeros)
  {
    float *total_ptr{nullptr};
    float *gaussian_ptr{nullptr};
    float *impulsive_ptr{nullptr};

    for (auto ichan = 0; ichan < nchan; ichan++)
    {
      for (auto ipol = 0; ipol < npol; ipol++)
      {
        total_ptr = source->get_output()->get_datptr(ichan, ipol);
        if (gaussian_noise)
        {
          gaussian_ptr = gaussian_noise->get_output()->get_datptr(ichan, ipol);
        }
        if (impulsive_noise)
        {
          impulsive_ptr = impulsive_noise->get_output()->get_datptr(ichan, ipol);
        }

        auto nval = ndat * ndim;
        for (auto ival = 0; ival < nval; ival++)
        {
          float actual_value = total_ptr[ival];
          if (all_zeros)
          {
            ASSERT_FLOAT_EQ(actual_value, 0.0)
              << "ichan=" << ichan
              << ", ipol=" << ipol
              << ", ival=" << ival;
          }
          else
          {
            float gaussian_value = (gaussian_ptr) ? gaussian_ptr[ival] : 0.0;
            float impulsive_value = (impulsive_ptr) ? impulsive_ptr[ival] : 0.0;
            float expected_value = gaussian_value + impulsive_value;

            ASSERT_FLOAT_EQ(actual_value, expected_value)
              << "ichan=" << ichan
              << ", ipol=" << ipol
              << ", ival=" << ival
              << ", gaussian_value=" << gaussian_value
              << ", impulsive_value=" << impulsive_value;
          }
        }
      }
    }
  }

  void SumSourceTest::assert_tfp(std::shared_ptr<SumSource> source, bool all_zeros)
  {
    float *total_ptr{nullptr};
    float *gaussian_ptr{nullptr};
    float *impulsive_ptr{nullptr};

    total_ptr = source->get_output()->get_dattfp();
    if (gaussian_noise)
    {
      gaussian_ptr = gaussian_noise->get_output()->get_dattfp();
    }
    if (impulsive_noise)
    {
      impulsive_ptr = impulsive_noise->get_output()->get_dattfp();
    }
    auto nval = ndat * nchan * npol * ndim;
    for (auto ival = 0; ival < nval; ival++)
    {
      float actual_value = total_ptr[ival];
      if (all_zeros)
      {
        ASSERT_FLOAT_EQ(actual_value, 0.0) << "ival=" << ival;
      }
      else
      {
        float gaussian_value = (gaussian_ptr) ? gaussian_ptr[ival] : 0.0;
        float impulsive_value = (impulsive_ptr) ? impulsive_ptr[ival] : 0.0;
        float expected_value = gaussian_value + impulsive_value;

        ASSERT_FLOAT_EQ(actual_value, expected_value)
          << "ival=" << ival
          << ", gaussian_value=" << gaussian_value
          << ", impulsive_value=" << impulsive_value;
      }
    }

  }

  void SumSourceTest::assert_data(std::shared_ptr<SumSource> source, bool all_zeros)
  {

    switch (order)
    {
    case dsp::TimeSeries::OrderFPT:
      assert_fpt(source, all_zeros);
      break;
    case dsp::TimeSeries::OrderTFP:
    default:
      assert_tfp(source, all_zeros);
      break;
    }

  }

  dsp::test::SumSource* SumSourceTest::source_under_test(int niterations)
  {
    Reference::To<dsp::test::SumSource> source = new dsp::test::SumSource(niterations);
    return source.release();
  }

  bool SumSourceTest::prepare_source(std::shared_ptr<dsp::test::SumSource> source)
  try
  {
    output->set_nchan(nchan);
    output->set_npol(npol);
    output->set_ndim(ndim);
    output->resize(ndat);
    output->set_rate(tsamp_sec);
    output->set_order(order);

    source->set_output(output);
    source->prepare();

    if (gaussian_noise)
    {
      source->add_source(gaussian_noise);
    }

    if (impulsive_noise)
    {
      impulsive_noise->set_height(height);
      impulsive_noise->set_period(period);
      impulsive_noise->set_impulse_duration(impulse_duration);
      impulsive_noise->set_phase_offset(phase_offset);
      source->add_source(impulsive_noise);
    }

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

  TEST_P(SumSourceTest, test_happy_path) // NOLINT
  {
    std::shared_ptr<dsp::test::SumSource> source(source_under_test());
    ASSERT_NE(source, nullptr);

    gaussian_noise = new dsp::test::GaussianNoiseSource;
    impulsive_noise = new dsp::test::ImpulsiveNoiseSource;

    prepare_source(source);

    ASSERT_NE(source->get_output(), nullptr);

    ASSERT_NE(gaussian_noise->get_output(), nullptr);
    ASSERT_EQ(gaussian_noise->get_output()->get_order(), order);
    ASSERT_EQ(gaussian_noise->get_output()->get_order(), source->get_output()->get_order());

    ASSERT_NE(impulsive_noise->get_output(), nullptr);
    ASSERT_EQ(impulsive_noise->get_output()->get_order(), order);
    ASSERT_EQ(impulsive_noise->get_output()->get_order(), source->get_output()->get_order());

    ASSERT_FALSE(source->end_of_data());
    ASSERT_FALSE(gaussian_noise->end_of_data());
    ASSERT_FALSE(impulsive_noise->end_of_data());
    source->operate();
    ASSERT_TRUE(source->end_of_data());
    ASSERT_TRUE(gaussian_noise->end_of_data());
    ASSERT_TRUE(impulsive_noise->end_of_data());

    assert_data(source);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  // TODO - happy multiple iterations path

  TEST_P(SumSourceTest, test_no_sources_produce_zeroed_data) // NOLINT
  {
    std::shared_ptr<dsp::test::SumSource> source(source_under_test());
    ASSERT_NE(source, nullptr);

    prepare_source(source);
    ASSERT_NE(source->get_output(), nullptr);

    ASSERT_FALSE(source->end_of_data());
    source->operate();
    ASSERT_TRUE(source->end_of_data());

    assert_data(source, true);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_P(SumSourceTest, test_gaussian_noise_only) // NOLINT
  {
    std::shared_ptr<dsp::test::SumSource> source(source_under_test());
    ASSERT_NE(source, nullptr);
    gaussian_noise = new dsp::test::GaussianNoiseSource;

    prepare_source(source);
    ASSERT_NE(source->get_output(), nullptr);
    ASSERT_NE(gaussian_noise->get_output(), nullptr);
    ASSERT_EQ(gaussian_noise->get_output()->get_order(), order);
    ASSERT_EQ(gaussian_noise->get_output()->get_order(), source->get_output()->get_order());

    ASSERT_FALSE(source->end_of_data());
    ASSERT_FALSE(gaussian_noise->end_of_data());
    source->operate();
    ASSERT_TRUE(source->end_of_data());
    ASSERT_TRUE(gaussian_noise->end_of_data());

    assert_data(source);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_P(SumSourceTest, test_impulsive_noise_only) // NOLINT
  {
    std::shared_ptr<dsp::test::SumSource> source(source_under_test());
    ASSERT_NE(source, nullptr);
    impulsive_noise = new dsp::test::ImpulsiveNoiseSource;

    prepare_source(source);
    ASSERT_NE(source->get_output(), nullptr);
    ASSERT_NE(impulsive_noise->get_output(), nullptr);
    ASSERT_EQ(impulsive_noise->get_output()->get_order(), order);
    ASSERT_EQ(impulsive_noise->get_output()->get_order(), source->get_output()->get_order());

    ASSERT_FALSE(source->end_of_data());
    ASSERT_FALSE(impulsive_noise->end_of_data());
    source->operate();
    ASSERT_TRUE(source->end_of_data());
    ASSERT_TRUE(impulsive_noise->end_of_data());

    assert_data(source);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  INSTANTIATE_TEST_SUITE_P(
      SumSourceTestSuite, SumSourceTest,
      testing::Values(dsp::TimeSeries::OrderTFP, dsp::TimeSeries::OrderFPT),
      [](const testing::TestParamInfo<SumSourceTest::ParamType> &info)
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
