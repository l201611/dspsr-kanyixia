/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GtestMain.h"
#include "dsp/TimeSeries.h"

#include "dsp/RescaleMedianMadCalculatorTest.h"

// from PSRCHIVE / EPSIC
#include "BoxMuller.h"

#include <iostream>
#include <algorithm>
#include <random>

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

RescaleMedianMadCalculatorTest::RescaleMedianMadCalculatorTest()
{
  input = new dsp::TimeSeries;
}

void RescaleMedianMadCalculatorTest::SetUp()
{
  double tsamp{0.000064};
  double day = 12345;
  double ss = 54;
  double fs = 0.222;
  MJD epoch(day, ss, fs);

  input->set_rate(1/tsamp);
  input->set_start_time(epoch);
}

void RescaleMedianMadCalculatorTest::TearDown()
{
}

void RescaleMedianMadCalculatorTest::generate_fpt()
{
  input->set_order(dsp::TimeSeries::OrderFPT);
  input->resize(ndat);

  std::vector<float> data(input->get_ndat() * input->get_ndim(), 0.0);
  time_t now = time(nullptr);
  BoxMuller bm(now);

  for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
  {
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      std::generate(data.begin(), data.end(), bm);

      float * ptr = input->get_datptr(ichan, ipol);
      uint64_t ival = 0;
      for (uint64_t idat=0; idat<input->get_ndat(); idat++)
      {
        for (unsigned idim=0; idim<input->get_ndim(); idim++)
        {
          ptr[ival] = (data[idat] * input_scale) + input_offset;
          ival++;
        }
      }
    }
  }
}

void RescaleMedianMadCalculatorTest::generate_tfp()
{
  input->set_order(dsp::TimeSeries::OrderTFP);
  input->resize(ndat);

  // generate the required number of normally distributed values with zero mean and unit variance
  size_t nval = ndat * input->get_ndim() * input->get_npol() * input->get_nchan();
  std::vector<float> data(nval, 0.0);
  time_t now = time(nullptr);
  BoxMuller bm(now);
  std::generate(data.begin(), data.end(), bm);

  float * ptr = input->get_dattfp();
  uint64_t ival = 0;

  for (unsigned idat=0; idat<ndat; idat++)
  {
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
      {
        for (unsigned idim=0; idim<input->get_ndim(); idim++)
        {
          ptr[ival] = (data[ival] * input_scale) + input_offset;
          ival++;
        }
      }
    }
  }
}

void RescaleMedianMadCalculatorTest::generate_data()
{
  // set the values here rather than SetUp to allow for tests to override nchan, npol and ndim
  input->set_nchan(nchan);
  input->set_npol(npol);
  input->set_ndim(ndim);

  if (order == dsp::TimeSeries::OrderFPT)
    generate_fpt();
  else
    generate_tfp();
}

void RescaleMedianMadCalculatorTest::assert_expected_statistics(std::vector<std::vector<std::vector<float>>> data)
{
  for (unsigned ipol=0; ipol<data.size(); ipol++)
  {
    for (unsigned ichan=0; ichan<data[ipol].size(); ichan++)
    {
      auto curr_data = data[ipol][ichan];
      auto nval = curr_data.size();
      if (nval == 0)
        continue;

      float sum{0.0}, sumsq{0.0};
      for (auto& val : curr_data)
      {
        sum += val;
        sumsq += (val * val);
      }
      float data_mean = sum / nval;
      float data_variance = (sumsq / nval - data_mean * data_mean);
      std::sort(curr_data.begin(), curr_data.end());

      auto median_idx = (curr_data.size() - 1) / 2;
      auto expected_median = curr_data[median_idx];

      for (auto idx = 0; idx < curr_data.size(); idx++)
      {
        curr_data[idx] = fabs(curr_data[idx] - expected_median);
      }

      std::sort(curr_data.begin(), curr_data.end());
      auto expected_mad = curr_data[median_idx];
      if (expected_mad == 0.0)
        expected_mad = 1.0;

      auto expected_std = expected_mad / scale_factor;
      auto expected_scale = 1.0 / expected_std;
      auto expected_variance = expected_std * expected_std;

      auto offset = calculator->get_offset(ipol)[ichan];
      auto scale = calculator->get_scale(ipol)[ichan];

      ASSERT_NE(data_mean, -offset);
      ASSERT_NE(data_variance, 1/(scale * scale));

      ASSERT_NEAR(offset, -expected_median, 1e-6)
        << "ipol=" << ipol << ", ichan=" << ichan;

      ASSERT_NEAR(scale, expected_scale, 1e-6)
        << "ipol=" << ipol << ", ichan=" << ichan;

      auto calc_mean = static_cast<float>(calculator->get_mean(ipol)[ichan]);
      auto calc_variance = static_cast<float>(calculator->get_variance(ipol)[ichan]);

      ASSERT_NEAR(calc_mean, expected_median, 1e-6)
        << "ipol=" << ipol << ", ichan=" << ichan;

      ASSERT_NEAR(calc_variance, expected_variance, 1e-6)
        << "ipol=" << ipol << ", ichan=" << ichan;
    }
  }
}

void RescaleMedianMadCalculatorTest::assert_fpt()
{
  if (dsp::Operation::verbose)
    std::cerr << "RescaleMedianMadCalculatorTest::assert_fpt - asserting FPT data" << std::endl;

  std::vector<std::vector<std::vector<float>>> data;
  data.resize(input->get_npol());

  uint64_t nval = input->get_ndat() * input->get_ndim();
  for (unsigned ipol=0; ipol < input->get_npol(); ipol++)
  {
    data[ipol].resize(input->get_nchan());
    for (unsigned ichan=0; ichan < input->get_nchan(); ichan++)
    {
      data[ipol][ichan].resize(nval, 0.0);
    }
  }

  for (unsigned ichan=0; ichan < input->get_nchan(); ichan++)
  {
    for (unsigned ipol=0; ipol < input->get_npol(); ipol++)
    {
      const float * ptr = input->get_datptr(ichan, ipol);
      for (uint64_t ival=0; ival < nval; ival++)
      {
        data[ipol][ichan][ival] = ptr[ival];
      }
    }
  }
  assert_expected_statistics(data);
}

void RescaleMedianMadCalculatorTest::assert_tfp()
{
  if (dsp::Operation::verbose)
    std::cerr << "RescaleMedianMadCalculatorTest::assert_tfp - asserting TFP data" << std::endl;

  std::vector<std::vector<std::vector<float>>> data;
  data.resize(input->get_npol());

  uint64_t nval = input->get_ndat() * input->get_ndim();
  for (unsigned ipol=0; ipol < input->get_npol(); ipol++)
  {
    data[ipol].resize(input->get_nchan());
    for (unsigned ichan=0; ichan < input->get_nchan(); ichan++)
    {
      data[ipol][ichan].resize(nval, 0.0);
    }
  }

  float * ptr = input->get_dattfp();
  for (unsigned idat=0; idat<ndat; idat++)
  {
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
      {
        uint64_t ival = idat * input->get_ndim();
        for (unsigned idim=0; idim<input->get_ndim(); idim++, ival++)
        {
          data[ipol][ichan][ival] = *ptr;
          ptr++;
        }
      }
    }
  }
  assert_expected_statistics(data);
}

void RescaleMedianMadCalculatorTest::assert_data()
{
  if (order == dsp::TimeSeries::OrderFPT)
    assert_fpt();
  else
    assert_tfp();
}

bool RescaleMedianMadCalculatorTest::perform_computation()
  try
{
  // ensure we have the correct scale factor for testing
  calculator->set_scale_factor(scale_factor);
  calculator->init(input, ndat, false);
  calculator->sample_data(input, 0, ndat, false);
  calculator->compute();

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

TEST_F(RescaleMedianMadCalculatorTest, test_construct_delete) // NOLINT
{
  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NE(calculator, nullptr);
  calculator = nullptr;
  ASSERT_EQ(calculator, nullptr);
}

TEST_P(RescaleMedianMadCalculatorTest, test_happy_path)
{
  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NO_THROW(perform_computation());

  assert_data();
}

TEST_P(RescaleMedianMadCalculatorTest, test_large_ndat)
{
  ndat = 32768;
  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NO_THROW(perform_computation());

  assert_data();
}

TEST_P(RescaleMedianMadCalculatorTest, test_large_nchan)
{
  nchan = 432;
  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NO_THROW(perform_computation());

  assert_data();
}

TEST_P(RescaleMedianMadCalculatorTest, test_npol)
{
  npol = 1;
  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NO_THROW(perform_computation());
  assert_data();

  npol = 2;
  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NO_THROW(perform_computation());
  assert_data();
}

TEST_P(RescaleMedianMadCalculatorTest, test_detected)
{
  npol = 4;
  ndim = 1;
  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NO_THROW(perform_computation());

  assert_data();
}

TEST_P(RescaleMedianMadCalculatorTest, test_sample_data_multiple_times)
{
  ndat = 32768;
  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NO_THROW(calculator->init(input, ndat, false));

  ASSERT_NO_THROW(calculator->sample_data(input, 0, ndat / 2, false));
  ASSERT_NO_THROW(calculator->sample_data(input, ndat/2, ndat, false));
  ASSERT_NO_THROW(calculator->compute());

  assert_data();
}

TEST_P(RescaleMedianMadCalculatorTest, test_scale_factor_applied)
{
  scale_factor = 1.0;
  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();

  auto default_scale_factor = dsp::RescaleMedianMadCalculator::DEFAULT_SCALE_FACTOR;
  ASSERT_EQ(calculator->get_scale_factor(), default_scale_factor);
  calculator->set_scale_factor(scale_factor);
  ASSERT_EQ(calculator->get_scale_factor(), scale_factor);

  ASSERT_NO_THROW(perform_computation());

  assert_data();
}

TEST_P(RescaleMedianMadCalculatorTest, test_random_input_scale_offset)
{
  std::random_device rd;
  std::mt19937 rand_engine(rd());
  std::uniform_real_distribution<> scale_dist(10.0, 100.0);
  std::uniform_real_distribution<> offset_dist(-1.0, 1.0);

  // have a larger ndat to ensure that calculated scale / offset are close
  ndat = 32768;
  input_scale = scale_dist(rand_engine);
  input_offset = offset_dist(rand_engine);

  order = GetParam();
  generate_data();

  calculator = std::make_shared<dsp::RescaleMedianMadCalculator>();
  ASSERT_NO_THROW(perform_computation());

  assert_data();
}

INSTANTIATE_TEST_SUITE_P(
    RescaleMedianMadCalculatorTestSuite, RescaleMedianMadCalculatorTest,
    testing::Values(dsp::TimeSeries::OrderFPT, dsp::TimeSeries::OrderTFP),
    [](const testing::TestParamInfo<RescaleMedianMadCalculatorTest::ParamType> &info)
    {
      auto order = info.param;

      std::string name;
      if (order == dsp::TimeSeries::OrderFPT)
        name = "fpt";
      else
        name = "tfp";

      return name;
    }); // NOLINT

} // namespace dsp::test
