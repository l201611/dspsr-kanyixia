/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/GaussianNoiseSourceTest.h"

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

  GaussianNoiseSourceTest::GaussianNoiseSourceTest()
  {
  }

  void GaussianNoiseSourceTest::SetUp()
  {
    if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
    {
      order = GetParam();
    }

    output = new dsp::TimeSeries;
    output->set_order(order);
  }

  void GaussianNoiseSourceTest::TearDown()
  {
    output = nullptr;
  }

  dsp::test::GaussianNoiseSource* GaussianNoiseSourceTest::source_under_test(int niterations)
  {
    Reference::To<dsp::test::GaussianNoiseSource> source = new dsp::test::GaussianNoiseSource(niterations);
    return source.release();
  }

  void GaussianNoiseSourceTest::assert_gaussian_noise(std::shared_ptr<GaussianNoiseSource> source, double threshold)
  {
    // get total sum & sum square + for channel + pol assert the variance of
    // expected values for the mean, stddev and variance after the Rescale/Digitizer operation
    std::vector<std::vector<double>> sums(nchan);
    std::vector<std::vector<double>> sum_sqs(nchan);

    for (unsigned ichan = 0; ichan < nchan; ichan++)
    {
      sums[ichan].resize(npol);
      sum_sqs[ichan].resize(npol);
      std::fill(sums[ichan].begin(), sums[ichan].end(), 0);
      std::fill(sum_sqs[ichan].begin(), sum_sqs[ichan].end(), 0);
    }

    switch (order)
    {
    case dsp::TimeSeries::OrderFPT:
      {
        for (unsigned ichan = 0; ichan < nchan; ichan++)
        {
          for (unsigned ipol = 0; ipol < npol; ipol++)
          {
            float *data_ptr = source->get_output()->get_datptr(ichan, ipol);
            unsigned nval = ndat * ndim;
            for (uint64_t ival = 0; ival < nval; ival++)
            {
              double val = static_cast<double>(data_ptr[ival]);
              sums[ichan][ipol] += val;
              sum_sqs[ichan][ipol] += val * val;
            }
          }
        }
      }
      break;
    case dsp::TimeSeries::OrderTFP:
    default:
      {
        float *data_ptr = source->get_output()->get_dattfp();
        unsigned ival = 0;
        for (uint64_t idat=0; idat < ndat; idat++)
        {
          for (unsigned ichan = 0; ichan < nchan; ichan++)
          {
            for (unsigned ipol = 0; ipol < npol; ipol++)
            {
              for (unsigned idim = 0; idim < ndim; idim++, ival++)
              {
                double val = static_cast<double>(data_ptr[ival]);
                sums[ichan][ipol] += val;
                sum_sqs[ichan][ipol] += val * val;
              }
            }
          }
        }
      }
      break;
    }

    double expected_variance = 1.0;
    double expected_mean = 0.0;
    double count = static_cast<double>(ndat * ndim);
    double error_of_mean = sqrt(expected_variance / count);

    // for a normal distribution, where mu_4 = 3 * variance^2
    double error_of_variance = sqrt((3.0 - double(count-3)/(count-1)) / count) * expected_variance;

    double assert_near_tolerance_mean = threshold * error_of_mean;
    double assert_near_tolerance_variance = threshold * error_of_variance;

    for (unsigned ichan = 0; ichan < nchan; ichan++)
    {
      for (unsigned ipol = 0; ipol < npol; ipol++)
      {
        double mean = sums[ichan][ipol] / count;
        double meansq = sum_sqs[ichan][ipol] / count;
        double variance = meansq - (mean * mean);

        ASSERT_NEAR(mean, expected_mean, assert_near_tolerance_mean);
        ASSERT_NEAR(variance, expected_variance, assert_near_tolerance_variance);
      }
    }
  }

  bool GaussianNoiseSourceTest::prepare_source(std::shared_ptr<dsp::test::GaussianNoiseSource> source)
  try
  {
    output->set_nchan(nchan);
    output->set_npol(npol);
    output->set_ndim(ndim);
    output->resize(ndat);
    output->set_rate(tsamp_sec);

    source->set_output(output);
    source->set_output_order(order);
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

  TEST_P(GaussianNoiseSourceTest, test_happy_path) // NOLINT
  {
    std::shared_ptr<dsp::test::GaussianNoiseSource> source(source_under_test());
    ASSERT_NE(source, nullptr);

    prepare_source(source);
    ASSERT_FALSE(source->end_of_data());
    source->operate();
    ASSERT_TRUE(source->end_of_data());

    assert_gaussian_noise(source);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  TEST_P(GaussianNoiseSourceTest, test_multiple_iterations) // NOLINT
  {
    std::shared_ptr<dsp::test::GaussianNoiseSource> source(source_under_test(2));
    ASSERT_NE(source, nullptr);

    prepare_source(source);
    ASSERT_FALSE(source->end_of_data());
    source->operate();
    ASSERT_FALSE(source->end_of_data());
    assert_gaussian_noise(source);

    source->operate();
    ASSERT_TRUE(source->end_of_data());
    assert_gaussian_noise(source);

    source = nullptr;
    ASSERT_EQ(source, nullptr);
  }

  INSTANTIATE_TEST_SUITE_P(
      GaussianNoiseSourceTestSuite, GaussianNoiseSourceTest,
      testing::Values(dsp::TimeSeries::OrderTFP, dsp::TimeSeries::OrderFPT),
      [](const testing::TestParamInfo<GaussianNoiseSourceTest::ParamType> &info)
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
