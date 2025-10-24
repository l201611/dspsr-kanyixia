/***************************************************************************
 *
 *   Copyright (C) 2025 by Andrew Jameson and Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TScrunchTest.h"
#include "dsp/TFPOffset.h"
#include "dsp/TimeSeries.h"
#include "dsp/GtestMain.h"
#include "dsp/Memory.h"
#include "dsp/SignalStateTestHelper.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <cassert>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_CUDA
#include "dsp/TScrunchCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#include <cuda.h>
#endif

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

TScrunchTest::TScrunchTest()
{
  states = { Signal::Intensity, Signal::PPQQ, Signal::Coherence };
  input = new dsp::TimeSeries;
  output = new dsp::TimeSeries;

#ifdef HAVE_CUDA
  /* CUDA-specific resources are constructed even if unused because
     the constructor is called only once */
  device_input = new dsp::TimeSeries;
  device_output = new dsp::TimeSeries;
  device_memory = new CUDA::DeviceMemory;
  device_input->set_memory(device_memory);
  device_output->set_memory(device_memory);
#endif
}

void TScrunchTest::SetUp()
{
  MJD::verbose = false;
  dsp::Observation::verbose = false;
  dsp::Operation::verbose = false;

  // configure temporal aspects of the time-series needed for Tscrunch
  static constexpr double tsamp = 0.000064;
  static constexpr double day = 12345;
  static constexpr double ss = 54;
  static constexpr double fs = 0.222;
  MJD epoch(day, ss, fs);

  std::cerr << "TScrunchTest::SetUp setting rate and start time" << std::endl;
  input->set_rate(1/tsamp);
  input->set_start_time(epoch);

  // TODO check what would be a better default
  input->set_input_sample(0);

  // TScrunch must operate on real-valued data
  static constexpr unsigned ndim = 1;
  input->set_ndim(ndim);
}

void TScrunchTest::TearDown()
{
  tscrunch_factor = 0;
}

unsigned TScrunchTest::generateRandomNumber(unsigned min, unsigned max)
{
  if (min > max)
  {
    throw std::runtime_error("TScrunchTest::generateRandomNumber min>max");
  }

  if (min == max)
  {
    return min;
  }

  // Seed the random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned> distribution(min, max);

  // Generate and return the random number
  return distribution(gen);
}

void TScrunchTest::generate_data()
{
  if (order == dsp::TimeSeries::OrderFPT)
    EXPECT_NO_THROW(generate_fpt());
  else
    EXPECT_NO_THROW(generate_tfp());
}

float TScrunchTest::get_expected_value(unsigned ichan, unsigned nchan, unsigned ipol, uint64_t odat, unsigned sfactor)
{
  float rate_of_change = static_cast<float>(ichan) / static_cast<float>(nchan);
  float offset = static_cast<float>(ipol);
  uint64_t idat = odat * sfactor;
  float expected = 0;
  for (unsigned isamp=0; isamp < sfactor; isamp++)
  {
    expected += (rate_of_change * static_cast<float>(idat + isamp)) + offset;
  }
  return expected;
}

void TScrunchTest::generate_fpt()
{
  input->set_order(dsp::TimeSeries::OrderFPT);
  output->set_order(dsp::TimeSeries::OrderFPT);

  input->set_input_sample(sample_number);
  input->resize(ndat);

  static constexpr unsigned sfactor = 1;
  for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
  {
    for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
    {
      float* ptr = input->get_datptr(ichan, ipol);
      assert(ptr != nullptr);

      uint64_t ival = 0;
      for (uint64_t idat=0; idat<input->get_ndat(); idat++)
      {
        ptr[ival] = get_expected_value(ichan, input->get_nchan(), ipol, sample_number + idat, sfactor);
        #ifdef _DEBUG
        std::cerr << "generate_fpt ipol=" << ipol << " ichan=" << ichan << " idat=" << (sample_number + idat) << " val=" << ptr[ival] << std::endl;
        #endif
        ival++;
      }
    }
  }
  sample_number += input->get_ndat();
}

void TScrunchTest::assert_data()
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
    output->zero();
    TransferCUDATestHelper xfer;
    xfer.copy(output, device_output, cudaMemcpyDeviceToHost);
  }
#endif

  if (dsp::Operation::verbose)
  {
    std::cerr << "TScrunchTest::assert_data input nchan=" << input->get_nchan() << " npol=" << input->get_npol() << " ndat=" << input->get_ndat() << std::endl;
    std::cerr << "TScrunchTest::assert_data output nchan=" << output->get_nchan() << " npol=" << output->get_npol() << " ndat=" << output->get_ndat() << std::endl;
    std::cerr << "TScrunchTest::assert_data verifying dimensions of output" << std::endl;
  }

  ASSERT_EQ(order, output->get_order());

  if (order == dsp::TimeSeries::OrderFPT)
    assert_fpt();
  else
    assert_tfp();

  if (dsp::Operation::verbose)
  {
    std::cerr << "TScrunchTest::assert_data all values are as expected" << std::endl;
  }
}

void TScrunchTest::assert_fpt()
{
  unsigned error_count = 0;
  unsigned error_limit = 0;

  uint64_t output_sample_number = output->get_input_sample();

  // frequency
  for (unsigned ichan=0; ichan<output->get_nchan(); ichan++)
  {
    // polarization
    for (unsigned ipol=0; ipol<output->get_npol(); ipol++)
    {
      const float* ptr = output->get_datptr(ichan, ipol);
      uint64_t ival = 0;

      for (uint64_t idat=0; idat<output->get_ndat(); idat++)
      {
        #ifdef _DEBUG
        std::cerr << "assert_fpt ipol=" << ipol << " ichan=" << ichan << " idat=" << (output_sample_number + idat) << " val=" << ptr[ival] << std::endl;
        #endif

        float expected = get_expected_value(ichan, output->get_nchan(), ipol, output_sample_number + idat, tscrunch_factor);
        float tolerance = (expected / 1e7) * tscrunch_factor;
        float abs_diff = fabs(ptr[ival] - expected);
        if (abs_diff > tolerance)
        {
          error_count ++;
          std::cerr << "TScrunchTest::assert_fpt FAIL ichan=" << ichan << " ipol=" << ipol << " idat=" << idat
                    << " val=" << ptr[ival] << " != " << expected << " abs_diff=" << abs_diff << " tolerance=" << tolerance << std::endl;
        }

        ival++;
      }
    }
  }

  ASSERT_EQ(error_count, error_limit);
}

void TScrunchTest::generate_tfp()
{
  input->set_order(dsp::TimeSeries::OrderTFP);
  output->set_order(dsp::TimeSeries::OrderTFP);

  input->set_input_sample(sample_number);
  input->resize(ndat);

  dsp::TFPOffset input_offset(input);

  static constexpr unsigned sfactor = 1;
  float* ptr = input->get_dattfp();
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
      {
        auto ival = input_offset(idat, ichan, ipol);
        ptr[ival] = get_expected_value(ichan, input->get_nchan(), ipol, sample_number + idat, sfactor);
        #ifdef _DEBUG
        std::cerr << "idat=" << idat << " ichan=" << ichan << " ipol=" << ipol << " ival=" << ival << " val=" << ptr[ival] << std::endl;
        #endif
      }
    }
  }

  sample_number += input->get_ndat();
}

void TScrunchTest::assert_tfp()
{
  ASSERT_EQ(dsp::TimeSeries::OrderTFP, output->get_order());

  TFPOffset output_offset(output);

  unsigned error_count = 0;
  unsigned error_limit = 0;
  const float* ptr = output->get_dattfp();
  uint64_t output_sample_number = output->get_input_sample();

  for (uint64_t idat=0; idat<output->get_ndat(); idat++)
  {
    for (unsigned ichan=0; ichan<output->get_nchan(); ichan++)
    {
      for (unsigned ipol=0; ipol<output->get_npol(); ipol++)
      {
        auto ival = output_offset(idat, ichan, ipol);
        float expected = get_expected_value(ichan, output->get_nchan(), ipol, output_sample_number + idat, tscrunch_factor);
        #ifdef _DEBUG
        std::cerr << "assert_tfp idat=" << output_sample_number + idat << " ichan=" << ichan << " ipol=" << ipol << " val=" << ptr[ival] << std::endl;
        #endif
        float tolerance = (expected / 1e7) * tscrunch_factor;
        float abs_diff = fabs(ptr[ival] - expected);
        if (abs_diff > tolerance)
        {
          error_count ++;
          std::cerr << "assert_tfp FAIL idat=" << idat << " ichan=" << ichan << " ipol=" << ipol
                    << " val=" << ptr[ival] << " != " << expected << std::endl;
        }
      }
    }
  }

  ASSERT_EQ(error_count, error_limit);
}

void TScrunchTest::assert_transform_configurations(std::shared_ptr<dsp::TScrunch> ts)
{
  ASSERT_EQ(tscrunch_factor, ts->get_factor());
}

bool TScrunchTest::perform_transform(std::shared_ptr<dsp::TScrunch> ts)
try
{
  if (!on_gpu)
  {
    ts->set_input(input);
    ts->set_output(output);
  }
#ifdef HAVE_CUDA
  else
  {
    TransferCUDATestHelper xfer;
    xfer.copy(device_input, input, cudaMemcpyHostToDevice);
    ts->set_input(device_input);
    ts->set_output(device_output);
  }
#endif

  ts->operate();
  return true;
}
catch (std::exception &exc) {
  std::cerr << "Exception Caught: " << exc.what() << std::endl;
  return false;
}
catch (Error &error)
{
  std::cerr << "Error Caught: " << error << std::endl;
  return false;
}

dsp::TScrunch* TScrunchTest::new_device_under_test()
{
  Reference::To<dsp::TScrunch> device = new dsp::TScrunch;
#ifdef HAVE_CUDA
  if (on_gpu)
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    device->set_engine(new CUDA::TScrunchEngine(stream));
  }
#endif
  return device.release();
}

TEST_P(TScrunchTest, test_construct_delete) // NOLINT
{
  std::shared_ptr<dsp::TScrunch> ts(new_device_under_test());
  ASSERT_NE(ts, nullptr);
  ts = nullptr;
  ASSERT_EQ(ts, nullptr);
}

TEST_P(TScrunchTest, test_tfp_small_large_nchanpoldim) // NOLINT
{
  auto param = GetParam();
  on_gpu = std::get<bool>(param);
  order = std::get<dsp::TimeSeries::Order>(param);

  static constexpr unsigned ntests = 4;

  // iterate through the possible polarisation states
  for (auto & state : states)
  {
    // iterate through the possible number of polarisations for the polarisation state
    std::vector<unsigned> npols = get_npols_for_state(state);
    for (auto & npol : npols)
    {
      std::cerr << "TScrunchTest::test_tfp_small_large_nchanpoldim"
        << " state=" << state_string(state)
        << " npol=" << npol
        << std::endl;

      input->set_state(state);
      input->set_npol(npol);

      // iterate a sequence of random scrunch factors
      for (unsigned itest=0; itest<ntests; itest++)
      {
        ndat = static_cast<uint64_t>(generateRandomNumber(ndat_range.first, ndat_range.second));

        // valid scrunch factors are between 2 and ndat
        tscrunch_factor = generateRandomNumber(2, static_cast<unsigned>(ndat));
        unsigned nblocks = generateRandomNumber(nblocks_range.first, nblocks_range.second);

        std::vector<unsigned> nchans(2);
        nchans[0] = generateRandomNumber(small_nchan_range.first, small_nchan_range.second);
        nchans[1] = generateRandomNumber(large_nchan_range.first, large_nchan_range.second);

        for (auto & nchan : nchans)
        {
          input->set_input_sample(0);
          input->set_nchan(nchan);
          if (dsp::Operation::verbose)
            std::cerr << "TScrunchTest::test_tfp_small_large_nchanpoldim nblocks=" << nblocks << " nchan=" << nchan << " ndat=" << ndat << " tscrunch_factor=" << tscrunch_factor << std::endl;

          sample_number = 0;
          std::shared_ptr<dsp::TScrunch> ts(new_device_under_test());

          // iterate through a sequence of blocks, to exercise input buffering
          for (unsigned iblock=0; iblock<nblocks; iblock++)
          {
            EXPECT_NO_THROW(generate_data());

            EXPECT_NO_THROW(ts->set_factor(tscrunch_factor));
            assert_transform_configurations(ts);
            ASSERT_TRUE(perform_transform(ts));
            assert_data();
          }
          ts.reset();
        } // for (nchans)
      } // for (ntests)
    } // for (npols)
  } // for (states)
}

TEST_P(TScrunchTest, test_sfactor_random) // NOLINT
{
  auto param = GetParam();
  on_gpu = std::get<bool>(param);
  order = std::get<dsp::TimeSeries::Order>(param);

  static constexpr unsigned ntests = 16;

  // iterate through the possible polarisation states
  for (auto & state : states)
  {
    // iterate through the possible number of polarisations for the polarisation state
    std::vector<unsigned> npols = get_npols_for_state(state);
    for (auto & npol : npols)
    {
      std::cerr << "TScrunchTest::test_sfactor_random"
        << " state=" << state_string(state)
        << " npol=" << npol
        << std::endl;

      input->set_state(state);
      input->set_npol(npol);

      // iterate a sequence of random scrunch factors
      for (unsigned itest=0; itest<ntests; itest++)
      {
        unsigned nchan = generateRandomNumber(nchan_range.first, nchan_range.second);
        ndat = static_cast<uint64_t>(generateRandomNumber(ndat_range.first, ndat_range.second));

        // valid scrunch factors are between 2 and ndat
        tscrunch_factor = generateRandomNumber(2, static_cast<unsigned>(ndat));
        unsigned nblocks = generateRandomNumber(nblocks_range.first, nblocks_range.second);

        input->set_input_sample(0);
        input->set_nchan(nchan);
        if (dsp::Operation::verbose)
          std::cerr << "TScrunchTest::test_sfactor_random nblocks=" << nblocks << " nchan=" << nchan << " ndat=" << ndat << " tscrunch_factor=" << tscrunch_factor << std::endl;

        sample_number = 0;
        std::shared_ptr<dsp::TScrunch> ts(new_device_under_test());

        // iterate through a sequence of blocks, to exercise input buffering
        for (unsigned iblock=0; iblock<nblocks; iblock++)
        {
          EXPECT_NO_THROW(generate_data());

          EXPECT_NO_THROW(ts->set_factor(tscrunch_factor));
          assert_transform_configurations(ts);
          ASSERT_TRUE(perform_transform(ts));
          assert_data();
        }
        ts.reset();
      }

    }
  }
}

TEST_P(TScrunchTest, test_very_large_nchan) // NOLINT
{
  auto param = GetParam();
  on_gpu = std::get<bool>(param);
  order = std::get<dsp::TimeSeries::Order>(param);

  unsigned nblocks = 2;
  ndat = 64;
  tscrunch_factor = 8;

  dsp::Operation::verbose = true;

  // iterate through the possible polarisation states
  for (auto & state : states)
  {
    // iterate through the possible number of polarisations for the polarisation state
    std::vector<unsigned> npols = get_npols_for_state(state);
    for (auto & npol : npols)
    {
      std::cerr << "TScrunchTest::test_sfactor_random"
        << " state=" << state_string(state)
        << " npol=" << npol
        << std::endl;

      input->set_state(state);
      input->set_npol(npol);

      unsigned nchan = generateRandomNumber(very_large_nchan_range.first, very_large_nchan_range.second);
      input->set_input_sample(0);
      input->set_nchan(nchan);
      if (dsp::Operation::verbose)
        std::cerr << "TScrunchTest::test_very_large_nchan nblocks=" << nblocks << " nchan=" << nchan << " ndat=" << ndat << " tscrunch_factor=" << tscrunch_factor << std::endl;

      sample_number = 0;
      std::shared_ptr<dsp::TScrunch> ts(new_device_under_test());

      // iterate through a sequence of blocks, to exercise input buffering
      for (unsigned iblock=0; iblock<nblocks; iblock++)
      {
        EXPECT_NO_THROW(generate_data());

        EXPECT_NO_THROW(ts->set_factor(tscrunch_factor));
        assert_transform_configurations(ts);
        ASSERT_TRUE(perform_transform(ts));
        assert_data();
      }
      ts.reset();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(TScrunchTestSuite, TScrunchTest,
  testing::Combine(testing::ValuesIn(get_gpu_flags()),
                   testing::Values(dsp::TimeSeries::OrderFPT, dsp::TimeSeries::OrderTFP)),
  [](const testing::TestParamInfo<TScrunchTest::ParamType>& info)
  {
    bool on_gpu = std::get<bool>(info.param);
    auto order = std::get<dsp::TimeSeries::Order>(info.param);
    std::string name;
    if (on_gpu)
      name = "on_gpu";
    else
      name = "on_cpu";
    if (order == dsp::TimeSeries::OrderFPT)
      name += "_fpt";
    else
      name += "_tfp";
    return name;
  }
); // NOLINT

} // namespace dsp::test
