/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos, Andrew Jameson, Will Gauvin and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/RescaleTest.h"
#include "dsp/RescaleMeanStdCalculator.h"
#include "dsp/RescaleMedianMadCalculator.h"

#include "dsp/GtestMain.h"
#include "dsp/TimeSeries.h"
#include "dsp/Memory.h"

#include "dsp/NormalSampleStats.h"
#include "dsp/SignalStateTestHelper.h"

// from PSRCHIVE / EPSIC
#include "BoxMuller.h"
#include "ascii_header.h"
#include "Estimate.h"
#include "dirutil.h"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_CUDA
#include "dsp/RescaleCUDA.h"
#include "dsp/RescaleMeanStdDevCalculatorCUDA.h"
#include "dsp/RescaleMedianMadCalculatorCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#endif

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <random>

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

RescaleTest::RescaleTest()
{
}

void RescaleTest::SetUp()
{
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

  if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
  {
    auto param = GetParam();
    on_gpu = param.on_gpu;
    order = param.order;
    mode_name = param.mode;
    dump_scale_offset = param.dump_scale_offset;
    state = param.state;
    npol = param.npol;
    calculator_name = param.calculator;
    if (param.dump_scale_offset)
    {
      scale_offset_dump_filepath = "/tmp/RescaleTest_scale_offset_dump.dada";
      scale_offset_dump = new dsp::RescaleScaleOffsetDump(scale_offset_dump_filepath);

      scale_offset_file_helper = new dsp::test::RescaleScaleOffsetDumpTestHelper;
    }
  }
  double tsamp{0.000064};
  double day = 12345;
  double ss = 54;
  double fs = 0.222;
  MJD epoch(day, ss, fs);

  input->set_rate(1/tsamp);
  input->set_nchan(nchan);
  input->set_npol(npol);
  input->set_start_time(epoch);
  input->set_ndim(ndim);
  input->set_input_sample(0);
  nsample = ndat;
}

void RescaleTest::TearDown()
{
  if (!scale_offset_dump_filepath.empty())
    std::remove(scale_offset_dump_filepath.c_str());

#ifdef HAVE_CUDA
  device_input = nullptr;
  device_output = nullptr;
  device_memory = nullptr;
#endif

  input = nullptr;
  output = nullptr;
  engine = nullptr;
  calculator = nullptr;
  scale_offset_dump = nullptr;
  scale_offset_file_helper = nullptr;
}

void RescaleTest::assert_metrics(std::shared_ptr<dsp::Rescale> rescale)
{
  if (rescale)
  {
    auto metrics = rescale->get_performance_metrics();
    ASSERT_TRUE(metrics->total_processing_time>0);
    ASSERT_TRUE(metrics->total_data_time>0);
    ASSERT_TRUE(metrics->total_bytes_processed>0);

    if (dsp::Operation::verbose)
    {
      std::cerr << "RescaleTest::perform_transform operate complete" << std::endl;
      std::cout << "Rescale Performance Metrics:" << std::endl;
      std::cout << "Total processing time: " << metrics->total_processing_time << " seconds" << std::endl;
      std::cout << "Total time spanned by the data: " << metrics->total_data_time << " seconds"  << std::endl;
      std::cout << "Total bytes processed: " << metrics->total_bytes_processed << std::endl;
    }
  }
  else
  {
    std::cerr << "No Rescale instance provided for metrics display." << std::endl;
  }
}

RescaleTest::Mode RescaleTest::mode_name_to_mode(std::string mode_name)
{
  if (mode_name == "HalfWave")
    return RescaleTest::Mode::HalfWave;
  else if (mode_name == "LinearRamp")
    return RescaleTest::Mode::LinearRamp;
  else if (mode_name == "AllZero")
    return RescaleTest::Mode::AllZero;
  else
    return RescaleTest::Mode::None;
}

void RescaleTest::get_offset_scale(RescaleTest::Mode mode, uint64_t idat, unsigned ipol, unsigned ichan, float *offset, float *scale)
{
  *offset = 0.1 + (0.1 * ipol);
  *scale = 2.0 + (1 * ichan);

  if (mode == HalfWave)
  {
    if (idat < ndat/2)
      *scale += 1;
    else
      *scale -= 1;
  }
  else if (mode == LinearRamp)
  {
    // range from -1 to +1 over the full ndat
    float factor = ((2 * static_cast<float>(idat)) / static_cast<float>(ndat)) - 1;
    *scale += factor;
  }
  else if (mode == AllZero)
  {
    *offset = 0.0;
    *scale = 0.0;
  }
}

void RescaleTest::generate_fpt()
{
  input->set_order(dsp::TimeSeries::OrderFPT);
  output->set_order(dsp::TimeSeries::OrderFPT);
#ifdef HAVE_CUDA
  if (on_gpu)
  {
    device_input->set_order(dsp::TimeSeries::OrderFPT);
    device_output->set_order(dsp::TimeSeries::OrderFPT);
  }
#endif

  if (dsp::Operation::verbose)
    std::cerr << "RescaleTest::generate_fpt input->resize(" << ndat << ")" << std::endl;
  input->resize(ndat);

  std::vector<float> data(input->get_ndat() * input->get_ndim(), 0.0);
  time_t now = time(nullptr);
  BoxMuller bm(now);

  float offset{0}, scale{0};
  for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
  {
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      std::generate(data.begin(), data.end(), bm);

      float * ptr = input->get_datptr(ichan, ipol);
      uint64_t ival = 0;
      for (uint64_t idat=0; idat<input->get_ndat(); idat++)
      {
        get_offset_scale(input_mode, idat, ipol, ichan, &offset, &scale);
        for (unsigned idim=0; idim<input->get_ndim(); idim++)
        {
          ptr[ival] = (data[idat] * scale) + offset;
          ival++;
        }
      }
    }
  }
}

void RescaleTest::generate_tfp()
{
  input->set_order(dsp::TimeSeries::OrderTFP);
  output->set_order(dsp::TimeSeries::OrderTFP);
#ifdef HAVE_CUDA
  if (on_gpu)
  {
    device_input->set_order(dsp::TimeSeries::OrderTFP);
    device_output->set_order(dsp::TimeSeries::OrderTFP);
  }
#endif

  input->resize(ndat);

  // generate the required number of normally distributed values with zero mean and unit variance
  size_t nval = ndat * input->get_ndim() * input->get_npol() * input->get_nchan();
  std::vector<float> data(nval, 0.0);
  time_t now = time(nullptr);
  BoxMuller bm(now);
  std::generate(data.begin(), data.end(), bm);

  float * ptr = input->get_dattfp();
  uint64_t ival = 0;
  float offset{0}, scale{0};
  for (unsigned idat=0; idat<ndat; idat++)
  {
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
      {
        get_offset_scale(input_mode, idat, ipol, ichan, &offset, &scale);
        for (unsigned idim=0; idim<input->get_ndim(); idim++)
        {
          ptr[ival] = (data[ival] * scale) + offset;
          ival++;
        }
      }
    }
  }
}

void RescaleTest::generate_data()
{
  if (order == dsp::TimeSeries::OrderFPT)
    EXPECT_NO_THROW(generate_fpt());
  else
    EXPECT_NO_THROW(generate_tfp());
}

void RescaleTest::assert_offsets_scales()
{
  if (dsp::Observation::verbose)
    std::cerr << "RescaleTest::assert_offsets_scales" << std::endl;

  // strong assertions can only be made when the input is normally distributed
  if ((mode_name == "HalfWave") || (mode_name == "AllZero") || (mode_name == "LinearRamp"))
  {
    if (dsp::Operation::verbose)
      std::cerr << "RescaleTest::assert_offsets_scales skipping tests as "
                << mode_name << " not normally distributed" << std::endl;
    return;
  }

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    output->zero();
    TransferCUDATestHelper xfer;
    xfer.copy(output, device_output, cudaMemcpyDeviceToHost);
  }
#endif

  float applied_offset{0}, applied_scale{0};
  uint64_t idat = 0;
  uint64_t count = std::min(output->get_ndat(), nsample) * output->get_ndim();
  NormalSampleStats stats;
  stats.set_ndat (count);

  for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
  {
    const float * offsets = rescale->get_offset(ipol);
    const float * scales = rescale->get_scale(ipol);
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      get_offset_scale(offsets_scales_mode, idat, ipol, ichan, &applied_offset, &applied_scale);

      float expected_offset = -applied_offset;
      float expected_variance = applied_scale * applied_scale;

      stats.set_variance (static_cast<double>(expected_variance));
      double error_of_offset = stats.get_sample_mean_stddev();

      Estimate<double> var;
      var.set_value(expected_variance);
      var.set_error(stats.get_sample_variance_stddev());

      Estimate<double> scale = 1.0 / sqrt(var);
      float expected_scale = static_cast<float>(scale.get_value());
      float error_of_scale = static_cast<float>(scale.get_error());

      double threshold = (calculator_name == "MedianMad") ? 8.0 : 6.0;
      float offset_tolerance = static_cast<float>(threshold * error_of_offset);
      float scale_tolerance = static_cast<float>(threshold * error_of_scale);

      if (dsp::Observation::verbose)
        std::cerr << "expected_offset=" << expected_offset
          << ", expected_variance=" << expected_variance
          << ", expected_scale=" << expected_scale
          << ", offset=" << offsets[ichan]
          << ", scale=" << scales[ichan]
          << std::endl;

      EXPECT_NEAR(offsets[ichan], expected_offset, offset_tolerance)
          << "count=" << count
          << ", ichan=" << ichan
          << ", ipol=" << ipol
          << ", threshold=" << threshold
          << ", error_of_offset=" << error_of_offset
          << ", sigma=" << fabs(offsets[ichan] - expected_offset) / error_of_offset;

      EXPECT_NEAR(scales[ichan], expected_scale, scale_tolerance)
          << "count=" << count
          << ", ichan=" << ichan
          << ", ipol=" << ipol
          << ", threshold=" << threshold
          << ", error_of_scale=" << error_of_scale
          << ", sigma=" << fabs(scales[ichan] - expected_scale) / error_of_scale;

    }
  }
}

void RescaleTest::assert_expected_rescaled_statistics(uint64_t nsample, const std::vector<std::vector<double>> &sums, const std::vector<std::vector<double>> &sums_sq)
{
  if (dsp::Observation::verbose)
    std::cerr << "RescaleTest::assert_expected_rescaled_statistics" << std::endl;

  // expected values for the mean, stddev and variance after the Rescale operation
  static constexpr double expected_rescaled_mean = 0.0;
  static constexpr double expected_rescaled_variance = 1.0;

  // determine the number of values over which the statistics can be asserted
  uint64_t output_ndat = std::min(output->get_ndat(), nsample);
  double nval = static_cast<double>(output_ndat * output->get_ndim());

  double error_of_mean = sqrt(expected_rescaled_variance / nval);

  // for a normal distribution, where mu_4 = 3 * variance^2
  double error_of_variance = sqrt((3.0 - double(nval-3)/(nval-1)) / nval) * expected_rescaled_variance;

  double threshold = (calculator_name == "MedianMad") ? 8.0 : 6.0; // sigma -> 3.4 false positives per million tests
  double assert_near_tolerance_mean = threshold * error_of_mean;
  double assert_near_tolerance_variance = threshold * error_of_variance;

  for (unsigned ichan=0; ichan<sums.size(); ichan++)
  {
    for (unsigned ipol=0; ipol<sums[ichan].size(); ipol++)
    {
      double mean = sums[ichan][ipol] / nval;
      double mean_sq = sums_sq[ichan][ipol] / nval;
      double variance = mean_sq - (mean * mean);

      ASSERT_NEAR(mean, expected_rescaled_mean, assert_near_tolerance_mean)
        << "nsample=" << nsample
        << ", ichan=" << ichan
        << ", ipol=" << ipol
        << ", threshold=" << threshold
        << ", error_of_mean=" << error_of_mean
        << ", sigma=" << fabs(mean - expected_rescaled_mean) / error_of_mean;

      // in cases where all data are zero, the rescale operation defines the variance as 1
      if (variance > 0)
      {
        ASSERT_NEAR(variance, expected_rescaled_variance, assert_near_tolerance_variance)
          << "nsample=" << nsample
          << ", ichan=" << ichan
          << ", ipol=" << ipol
          << ", threshold=" << threshold
          << ", error_of_variance=" << error_of_variance
          << ", sigma=" << fabs(variance - expected_rescaled_variance) / error_of_variance;
      }
    }
  }
}

void RescaleTest::assert_fpt()
{
  if (dsp::Observation::verbose)
    std::cerr << "RescaleTest::assert_fpt" << std::endl;

  // compute the sum and sumsq for each channel/pol
  std::vector<std::vector<double>> sums(output->get_nchan());
  std::vector<std::vector<double>> sums_sq(output->get_nchan());
  for (unsigned ichan=0; ichan<output->get_nchan(); ichan++)
  {
    sums[ichan].resize(output->get_npol(), 0);
    sums_sq[ichan].resize(output->get_npol(), 0);
  }

  uint64_t output_ndat = std::min(output->get_ndat(), nsample);
  for (unsigned ipol=0; ipol<output->get_npol(); ipol++)
  {
    for (unsigned ichan=0; ichan<output->get_nchan(); ichan++)
    {
      float * ptr = output->get_datptr(ichan, ipol);
      uint64_t ival = 0;
      for (uint64_t idat=0; idat<output_ndat; idat++)
      {
        for (unsigned idim=0; idim<output->get_ndim(); idim++)
        {
          const double val = static_cast<double>(ptr[ival]);
          sums[ichan][ipol] += val;
          sums_sq[ichan][ipol] += (val * val);
          ival++;
        }
      }
    }
  }
  assert_expected_rescaled_statistics(output_ndat, sums, sums_sq);
}

void RescaleTest::assert_tfp()
{
  if (dsp::Observation::verbose)
    std::cerr << "RescaleTest::assert_tfp" << std::endl;

  float * ptr = output->get_dattfp();

  // compute the sum and sumsq for each channel/pol
  std::vector<std::vector<double>> sums(output->get_nchan());
  std::vector<std::vector<double>> sums_sq(output->get_nchan());
  for (unsigned ichan=0; ichan<output->get_nchan(); ichan++)
  {
    sums[ichan].resize(output->get_npol(), 0);
    sums_sq[ichan].resize(output->get_npol(), 0);
  }

  uint64_t output_ndat = std::min(output->get_ndat(), nsample);
  for (unsigned idat=0; idat<output_ndat; idat++)
  {
    for (unsigned ichan=0; ichan<output->get_nchan(); ichan++)
    {
      for (unsigned ipol=0; ipol<output->get_npol(); ipol++)
      {
        for (unsigned idim=0; idim<output->get_ndim(); idim++)
        {
          const double val = static_cast<double>(*ptr);
          sums[ichan][ipol] += val;
          sums_sq[ichan][ipol] += (val * val);
          ptr++;
        }
      }
    }
  }
  assert_expected_rescaled_statistics(output_ndat, sums, sums_sq);
}

void RescaleTest::assert_data()
{
  if (dsp::Observation::verbose)
    std::cerr << "RescaleTest::assert_data" << std::endl;
#ifdef HAVE_CUDA
  if (on_gpu)
  {
    output->zero();
    TransferCUDATestHelper xfer;
    xfer.copy(output, device_output, cudaMemcpyDeviceToHost);
  }
#endif

  if (order == dsp::TimeSeries::OrderFPT)
    assert_fpt();
  else
    assert_tfp();

  if (dsp::Operation::verbose)
    std::cerr << "RescaleTest::assert_data all values are as expected" << std::endl;
}

void RescaleTest::assert_scale_offset_dump()
{
  if (dsp::Observation::verbose)
    std::cerr << "RescaleTest::assert_scale_offset_dump" << std::endl;
  scale_offset_file_helper->assert_file(scale_offset_dump_filepath);
}

bool RescaleTest::perform_transform() try
{
  if (!on_gpu)
  {
    rescale->set_input(input);
    rescale->set_output(output);
  }
#ifdef HAVE_CUDA
  else
  {
    TransferCUDATestHelper xfer;
    xfer.copy(device_input, input, cudaMemcpyHostToDevice);
    rescale->set_input(device_input);
    rescale->set_output(device_output);
  }
#endif

  if (nsample > 0)
  {
    rescale->set_interval_samples(nsample);
    if (use_exact_samples)
      rescale->set_exact(use_exact_samples);
  }
  rescale->set_constant(use_constant_thresholds);

  if (dsp::Operation::verbose)
    std::cerr << "RescaleTest::perform_transform exact=" << use_exact_samples << " constant=" << use_constant_thresholds << " nsample=" << nsample << std::endl;

  rescale->prepare();
  rescale->operate();

  assert_metrics(rescale);
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

void RescaleTest::new_transform_under_test()
{
  Reference::To<dsp::Rescale> rs = new dsp::Rescale;

  // set up calculator
  if (!on_gpu)
  {
    if (calculator_name == "MedianMad")
    {
      calculator = new dsp::RescaleMedianMadCalculator;
    }
    else
    {
      calculator = new dsp::RescaleMeanStdCalculator;
    }
    rs->set_calculator(calculator);
  }
#ifdef HAVE_CUDA
  else
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    engine = new CUDA::RescaleEngine(stream);
    if (calculator_name == "MeanStd")
    {
      calculator = new CUDA::RescaleMeanStdDevCalculatorCUDA(stream);
    }
    else
    {
      calculator = new CUDA::RescaleMedianMadCalculatorCUDA(stream);
    }
    rs->set_calculator(calculator);
    rs->set_engine(engine);
  }
#endif // HAVE_CUDA

  if (dump_scale_offset)
  {
    rs->add_callback_handler(scale_offset_dump, &dsp::RescaleScaleOffsetDump::handle_scale_offset_updated);
    rs->add_callback_handler(scale_offset_file_helper, &dsp::test::RescaleScaleOffsetDumpTestHelper::rescale_update);
  }

  rescale = std::shared_ptr<dsp::Rescale>(rs.release());

  if (!on_gpu)
  {
    rescale->set_input(input);
    rescale->set_output(output);
  }
#ifdef HAVE_CUDA
  else
  {
    TransferCUDATestHelper xfer;
    xfer.copy(device_input, input, cudaMemcpyHostToDevice);
    rescale->set_input(device_input);
    rescale->set_output(device_output);
  }
#endif
}

TEST_F(RescaleTest, test_construct_delete) // NOLINT
{
  new_transform_under_test();
  ASSERT_NE(rescale, nullptr);
  rescale = nullptr;
  ASSERT_EQ(rescale, nullptr);
}

/*
  The following test verifies tha the engine will have its calculator set
  if Rescale::set_calculator is called before Rescale::set_engine.
*/
TEST_F(RescaleTest, test_calculator_set_before_engine) // NOLINT
{
  Rescale rescale;

  MockCalculator calculator;
  rescale.set_calculator(&calculator);

  MockEngine engine;
  rescale.set_engine(&engine);

  ASSERT_EQ(rescale.get_calculator(), &calculator);
  ASSERT_EQ(engine.get_calculator(), &calculator);
}

/*
  The following test verifies tha the engine will have its calculator set
  if Rescale::set_calculator is called after Rescale::set_engine.
*/
TEST_F(RescaleTest, test_engine_set_before_calculator) // NOLINT
{
  Rescale rescale;

  MockEngine engine;
  rescale.set_engine(&engine);

  MockCalculator calculator;
  rescale.set_calculator(&calculator);

  ASSERT_EQ(rescale.get_calculator(), &calculator);
  ASSERT_EQ(engine.get_calculator(), &calculator);
}

TEST_P(RescaleTest, test_rescale_operation) // NOLINT
{
  offsets_scales_mode = input_mode = mode_name_to_mode(mode_name);
  if (input_mode == RescaleTest::Mode::LinearRamp || input_mode == RescaleTest::Mode::HalfWave)
    offsets_scales_mode = RescaleTest::Mode::None;

  ndim = get_ndim_for_state(state);

  input->set_state(state);
  input->set_npol(npol);
  input->set_ndim(ndim);
  nsample = ndat;
  new_transform_under_test();

  // the scale offset dump file should not exist before any operation is performed
  ASSERT_FALSE(file_exists(scale_offset_dump_filepath.c_str()));

  if (calculator_name == "MedianMad")
    test_using_median_mad();
  else
    test_using_mean_std();

  if (dump_scale_offset)
    assert_scale_offset_dump();
}

TEST_P(RescaleTest, test_rescale_nsample_range) // NOLINT
try {
  offsets_scales_mode = input_mode = mode_name_to_mode(mode_name);
  if (input_mode == RescaleTest::Mode::LinearRamp || input_mode == RescaleTest::Mode::HalfWave)
  {
    if (dsp::Operation::verbose)
      std::cerr << "RescaleTest::test_rescale_nsample_range skipping test with input_mode=" << input_mode << std::endl;
    return;
  }

  ndim = get_ndim_for_state(state);

  input->set_state(state);
  input->set_npol(npol);
  input->set_ndim(ndim);
  new_transform_under_test();

  // set the number of time samples to be a random number between ndat/2 and ndat
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> distribution((ndat/2)+1, ndat-1);
  nsample = distribution(gen);

  use_exact_samples = true;
  use_constant_thresholds = true;

  if (calculator_name == "MedianMad")
    test_using_median_mad();
  else
    test_using_mean_std();

  if (dump_scale_offset)
    assert_scale_offset_dump();
}
catch (Error& error)
{
  std::cerr << "RescaleTest, test_rescale_nsample_range exception " << error << std::endl;
  throw error += "RescaleTest, test_rescale_nsample_range";
}

TEST_P(RescaleTest, test_rescale_nsample_greater_than_ndat) // NOLINT
{
  offsets_scales_mode = input_mode = mode_name_to_mode(mode_name);
  if (input_mode == RescaleTest::Mode::LinearRamp || input_mode == RescaleTest::Mode::HalfWave)
  {
    if (dsp::Operation::verbose)
      std::cerr << "RescaleTest::test_rescale_nsample_greater_than_ndat skipping test with input_mode=" << input_mode << std::endl;
    return;
  }

  ndim = get_ndim_for_state(state);

  input->set_state(state);
  input->set_npol(npol);
  input->set_ndim(ndim);
  new_transform_under_test();

  // set the number of samples to be between ndat and thrice ndat
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> distribution(ndat+1, 3*ndat);
  nsample = distribution(gen);
  use_exact_samples = false;
  use_constant_thresholds = false;

  if (calculator_name == "MedianMad")
    test_using_median_mad();
  else
    test_using_mean_std();

  if (dump_scale_offset)
    assert_scale_offset_dump();
}

void RescaleTest::test_using_mean_std()
{
  if (dsp::Observation::verbose)
    std::cerr << "RescaleTest::test_using_mean_std mode=" << mode_name << " state=" << state_string(state) << " npol=" << npol << " ndim=" << ndim << std::endl;

  ScalesUpdatedMock callback_mock;
  unsigned expected_update_calls = nsample / ndat;
  if (nsample != ndat)
    expected_update_calls++;
  if (dsp::Operation::verbose)
    std::cerr << "RescaleTest::test_using_mean_std nsample=" << nsample << " expected_update_calls=" << expected_update_calls << std::endl;
  const auto *calculator = rescale->get_calculator();
  rescale->add_callback_handler(&callback_mock, &ScalesUpdatedMock::scales_updated);

  ASSERT_TRUE(calculator->get_scales().empty());
  ASSERT_TRUE(calculator->get_offsets().empty());
  EXPECT_CALL(callback_mock, scales_updated(testing::_)).Times(expected_update_calls);

  for (unsigned isample=0; isample < nsample; isample += ndat)
  {
    if (dsp::Observation::verbose)
    {
      std::cerr << "RescaleTest::test_using_mean_std executing generate/perform/assert loop for isample=" << isample << " nsample=" << nsample << " ndat=" << ndat << std::endl;
    }
    generate_data();
    ASSERT_TRUE(perform_transform());
    assert_offsets_scales();
    assert_data();
  }

  rescale->remove_callback_handler(&callback_mock, &ScalesUpdatedMock::scales_updated);
}

auto RescaleTest::expected_calls_of_scales_updated(uint64_t isample) -> int
{
  if (use_exact_samples)
  {
    //  if exact and constant - only 1 call and on the first loop, else 1 call per loop
    return use_constant_thresholds ? (isample == 0) : 1;
  }

  if (use_constant_thresholds)
  {
    // we need to perform a calculation per input_ndat until we reach nsample samples
    return isample < nsample;
  }

  // by the end of the iteration isample+ndat time samples would have been
  // sampled.  This would equate to floor((isample + ndat) / nsample) calculations
  // that have been performed.  We need to subtract the previous number of calculations
  // to get the expected number of calculations for this iteration.
  int previous_calculations = isample / nsample;
  int total_calculations = (isample + ndat) / nsample;

  auto expected_calculations = total_calculations - previous_calculations;

  // Until we have at least 1 computation (i.e. isample >= nsample) then in each
  // iteration of the transform we expect to perform a calculation, so that all the
  // ndat of the iteration are rescaled.
  if (expected_calculations == 0 && isample < nsample)
    expected_calculations++;

  if (dsp::Observation::verbose)
    std::cerr << "RescaleTest::expected_calls_of_scales_updated - "
      << "isample=" << isample
      << ", ndat=" << ndat
      << ", nsample=" << nsample
      << ", previous_calculations=" << previous_calculations
      << ", total_calculations=" << total_calculations
      << ", expected_calculations=" << expected_calculations
      << std::endl;

  return expected_calculations;
}

void RescaleTest::test_using_median_mad()
{
  // set up callback mock
  ScalesUpdatedMock callback_mock;
  rescale->add_callback_handler(&callback_mock, &ScalesUpdatedMock::scales_updated);

  ASSERT_TRUE(calculator->get_scales().empty());
  ASSERT_TRUE(calculator->get_offsets().empty());

  for (unsigned isample=0; isample < nsample; isample += ndat)
  {
    if (dsp::Observation::verbose)
    {
      std::cerr << "RescaleTest::test_using_median_mad executing generate/perform/assert loop for isample=" << isample << " nsample=" << nsample << " ndat=" << ndat << std::endl;
    }

    generate_data();

    auto expected_calls = expected_calls_of_scales_updated(isample);

    // this should give us the expected number of calls
    EXPECT_CALL(callback_mock, scales_updated(testing::_)).Times(expected_calls);

    ASSERT_TRUE(perform_transform());

    ASSERT_EQ(calculator->get_scales().size(), npol);
    ASSERT_EQ(calculator->get_offsets().size(), npol);

    assert_offsets_scales();
    assert_data();

    // assert data
    for (unsigned ipol = 0; ipol < npol; ipol++)
    {
      // asserts we have the right number of nchan for the pol
      ASSERT_EQ(calculator->get_scales()[ipol].size(), nchan);
      ASSERT_EQ(calculator->get_offsets()[ipol].size(), nchan);

      auto scales = calculator->get_scale(ipol);
      auto offsets = calculator->get_offset(ipol);
      ASSERT_TRUE(scales != nullptr);
      ASSERT_TRUE(offsets != nullptr);

      for (unsigned ichan = 0; ichan < nchan; ichan++)
      {
        ASSERT_NE(scales[ichan], 1.0);
        ASSERT_NE(offsets[ichan], 0.0);
      }
    }
  }

  // disconnect the callback_mock from the rescale callback
  rescale->remove_callback_handler(&callback_mock, &ScalesUpdatedMock::scales_updated);
}

std::vector<Signal::State> get_states()
{
  return { Signal::Nyquist, Signal::Analytic, Signal::Intensity, Signal::PPQQ, Signal::Coherence };
}

std::vector<std::string> get_calculators_for_device(bool on_gpu)
{
  return {"MeanStd", "MedianMad"};
}

std::vector<std::string> get_modes_for_calculator(std::string calculator)
{
  if (calculator == "MedianMad")
    return {"None"};
  else
    return {"None", "HalfWave", "LinearRamp", "AllZero"};
}

std::vector<dsp::test::TestParam> get_test_params()
{
  std::vector<dsp::test::TestParam> params{};

  for (auto on_gpu : get_gpu_flags())
  {
    std::vector<bool> dump_scale_offset_options{true, false};

    for (auto calculator : get_calculators_for_device(on_gpu))
    {
      for (auto mode : get_modes_for_calculator(calculator))
      {
        for (auto order : {dsp::TimeSeries::OrderFPT, dsp::TimeSeries::OrderTFP})
        {
          for (auto dump : dump_scale_offset_options)
          {
            for (auto state : get_states())
            {
              for (auto npol : get_npols_for_state(state))
              {
                params.push_back({ on_gpu, mode, order, calculator, state, npol, dump});
              }
            }
          }
        }
      }
    }
  }

  return params;
}

INSTANTIATE_TEST_SUITE_P(
    RescaleTestSuite, RescaleTest,
    testing::ValuesIn(get_test_params()),
    [](const testing::TestParamInfo<RescaleTest::ParamType> &info)
    {
      auto param = info.param;

      std::string name;
      if (param.on_gpu)
        name = "on_gpu";
      else
        name = "on_cpu";

      if (param.order == dsp::TimeSeries::OrderFPT)
        name += "_fpt_";
      else
        name += "_tfp_";

      if (param.calculator == "MeanStd")
        name += "mean_std_";
      else
        name += "median_mad_";

      name += (param.mode + "_");

      name += State2string(param.state);
      name += "_npol";
      name += std::to_string(param.npol);

      if (param.dump_scale_offset)
        name += "_dump";
      else
        name += "_nodump";

      return name;
    }); // NOLINT

} // namespace dsp::test
