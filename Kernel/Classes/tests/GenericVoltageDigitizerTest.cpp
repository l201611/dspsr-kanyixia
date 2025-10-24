/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/GenericVoltageDigitizerTest.h"

#include "dsp/Detection.h"
#include "dsp/GtestMain.h"
#include "dsp/TFPOffset.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "true_math.h"


#if HAVE_CUDA
#include "dsp/GenericVoltageDigitizerCUDA.h"
#include <cuda.h>
#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#endif

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

static constexpr int BITS_PER_BYTE = 8;

static constexpr int8_t tfp_expected_output_nbit1[] = {-94, -44, 5, -22};

static constexpr int8_t tfp_expected_output_nbit2[] = {
    -90, 98, -101, 92, -104, -86, 102, 70};

static constexpr int8_t tfp_expected_output_nbit4[] = {
    120, -72, 40, 121, -114, -109, -30, 119, -127, -77, -116,
    -119, 120, 56, 104, 50};

static constexpr int8_t tfp_expected_output_nbit8[] = {
    -128, 127, -51, -15, -26, 7, -22, 23, -5, -75, 12, -21, 7,
    -6, 29, 41, 4, -33, 12, -15, -12, -50, -20, -37, -23, 29,
    -67, 12, -23, 20, 9, 12};

static constexpr int16_t tfp_expected_output_nbit16[] = {
    -14132, 13977, -5538, -1643, -2814, 820, -2389, 2623, -445,
    -8168, 1399, -2249, 832, -626, 3271, 4558, 525, -3579, 1367,
    -1640, -1255, -5456, -2183, -4017, -2477, 3184, -7318, 1406,
    -2500, 2290, 1056, 1365};

static constexpr float tfp_expected_output_float[] = {
    -12.77228, 12.6336634, -5.005, -1.4842639, -2.5428553,
    0.7414923, -2.1585503, 2.3708253, -0.40148783, -7.3821115,
    1.2647325, -2.0321102, 0.7521466, -0.5648981, 2.9565287,
    4.120244, 0.47453195, -3.2339506, 1.2360413, -1.4815402,
    -1.1341237, -4.9306426, -1.9723871, -3.630312, -2.237967,
    2.8785143, -6.6134176, 1.2712177, -2.2587833, 2.0703325,
    0.95531327, 1.2340846};

static constexpr int8_t fpt_expected_output_nbit1[] = {
    -110, -108, -122, -50};

static constexpr int8_t fpt_expected_output_nbit2[] = {
    -74, 104, -102, 105, -62, 106, 86, 90};

static constexpr int8_t fpt_expected_output_nbit4[] = {
    120, -114, -127, 120, -72, -109, -77, 56, 40, -30, -116, 104, 121, 119, -119, 50};

static constexpr int8_t fpt_expected_output_nbit8[] = {
    -128, 127, -5, -75, 4, -33, -23, 29, -51, -15, 12,
    -21, 12, -15, -67, 12, -26, 7, 7, -6, -12, -50,
    -23, 20, -22, 23, 29, 41, -20, -37, 9, 12};

static constexpr int16_t fpt_expected_output_nbit16[] = {
    -14132, 13977, -445, -8168, 525, -3579, -2477, 3184, -5538,
    -1643, 1399, -2249, 1367, -1640, -7318, 1406, -2814, 820,
    832, -626, -1255, -5456, -2500, 2290, -2389, 2623, 3271,
    4558, -2183, -4017, 1056, 1365};

static constexpr int16_t fpt_expected_output_weights[] = {
    1, 1, 1, 1, 2, 2, 2, 2};

static constexpr float fpt_expected_output_float[] = {
    -12.77228, 12.6336634, -0.40148783, -7.3821115, 0.47453195,
    -3.2339506, -2.237967, 2.8785143, -5.005, -1.4842639, 1.2647325,
    -2.0321102, 1.2360413, -1.4815402, -6.6134176, 1.2712177,
    -2.5428553, 0.7414923, 0.7521466, -0.5648981, -1.1341237,
    -4.9306426, -2.2587833, 2.0703325, -2.1585503, 2.3708253,
    2.9565287, 4.120244, -1.9723871, -3.630312, 0.95531327, 1.2340846};

namespace dsp::test
{

  GenericVoltageDigitizerTest::GenericVoltageDigitizerTest()
  {
    std::cerr << "GenericVoltageDigitizerTest ctor" << std::endl;
  }

  void GenericVoltageDigitizerTest::SetUp()
  {


    if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
    {
      auto param = GetParam();
      nbit = std::get<0>(param);
      order = std::get<1>(param);
      on_gpu = std::get<2>(param);
      use_wts = std::get<3>(param);
    }
    else
    {
      use_wts = false;
    }

    if (use_wts)
    {
      input = input_wts = new dsp::WeightedTimeSeries;
      output_wts = new dsp::BitSeries;
    }
    else
    {
      input = new dsp::TimeSeries;
      input_wts = nullptr;
      output_wts = nullptr;
    }

    output = new dsp::BitSeries;
#ifdef HAVE_CUDA
    if (on_gpu)
    {
      if (dsp::Operation::verbose)
      {
        std::cerr << "SetUp on gpu" <<  std::endl;
      }
      cudaError_t result = cudaStreamCreate(&stream);
      ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);

      device_memory = new CUDA::DeviceMemory(stream);
      if (use_wts)
      {
        device_input = device_input_wts = new dsp::WeightedTimeSeries;
        device_input_wts->set_memory(device_memory);
        device_input_wts->set_weights_memory(device_memory);

        device_output_wts = new dsp::BitSeries;
        device_output_wts->set_memory(device_memory);
      }
      else
      {
        device_input = new TimeSeries;
        device_input->set_memory(device_memory);
      }
      device_input->set_order(order);
      device_output = new dsp::BitSeries;
      device_output->set_memory(device_memory);
    }
#endif
  }

  void GenericVoltageDigitizerTest::TearDown()
  {
    input = nullptr;
    output = nullptr;
    input_wts = nullptr;
    output_wts = nullptr;
#ifdef HAVE_CUDA
    if (on_gpu)
    {
      cudaError_t result = cudaStreamSynchronize(stream);
      ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);
      result = cudaStreamDestroy(stream);
      if (result != cudaSuccess)
        std::cerr << "cudaStreamDestroy: " << cudaGetErrorString(result) << std::endl;
      ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);
    }

    device_input = nullptr;
    device_input_wts = nullptr;
    device_output = nullptr;
    device_output_wts = nullptr;
    device_memory = nullptr;
#endif
  }

  void GenericVoltageDigitizerTest::assert_metrics(std::shared_ptr<dsp::GenericVoltageDigitizer> gvd)
  {
    if (gvd)
    {
      auto metrics = gvd->get_performance_metrics();
      if (dsp::Operation::verbose)
      {
        std::cout << "GenericVoltageDigitizer Performance Metrics:" << std::endl;
        std::cout << "Total Processing Time: " << metrics->total_processing_time << " seconds" << std::endl;
        std::cout << "Total time spanned by the data: " << metrics->total_data_time << " seconds"  << std::endl;
        std::cout << "Total Bytes Processed: " << metrics->total_bytes_processed << std::endl;
      }
      ASSERT_TRUE(metrics->total_processing_time >= 0); // may need nanosecond precision for tests that are too fast
      ASSERT_TRUE(metrics->total_data_time > 0);
      ASSERT_TRUE(metrics->total_bytes_processed > 0);
    }
    else
    {
      std::cerr << "No GenericVoltageDigitizer instance provided for metrics display." << std::endl;
    }
  }

  dsp::GenericVoltageDigitizer* GenericVoltageDigitizerTest::new_device_under_test()
  {
    Reference::To<dsp::GenericVoltageDigitizer> device = new dsp::GenericVoltageDigitizer;
  #ifdef HAVE_CUDA
    if (on_gpu)
    {
      if (dsp::Operation::verbose)
      {
        std::cerr << "GenericVoltageDigitizerTest::new_device_under_test CUDA::GenericVoltageDigitizerEngine" << std::endl;
      }
      device->set_device(device_memory);
    }
  #endif
    return device.release();
  }

  // Function to generate a normally generated data with a given scale and mean
  void GenericVoltageDigitizerTest::generate_random_data(float *data_ptr, size_t data_size, float scale, float mean)
  {
    // Seed the random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::mt19937 gen(seed);
    std::normal_distribution<float> distribution(mean, scale);

    for (uint64_t i = 0; i < data_size; i++)
    {
      data_ptr[i] = distribution(gen);
    }
  }

  void GenericVoltageDigitizerTest::setup_zeroed_data()
  {
    output->zero();
    input->set_state(Signal::Analytic);
    input->set_order(order);
    input->set_ndim(ndim);
    input->set_npol(npol);
    input->set_nchan(nchan);
    input->set_rate(1.0);
    if (use_wts)
    {
      input_wts->resize(ndat);
      input_wts->set_ndat_per_weight(ndat_per_weight);
      generate_weights();
    }
    else
    {
      input->resize(ndat);
    }

    float *data_ptr{nullptr};
    switch (order)
    {
    case dsp::TimeSeries::OrderFPT:
      data_ptr = input->get_datptr();
      break;

    case dsp::TimeSeries::OrderTFP:
    default:
      data_ptr = input->get_dattfp();
      break;
    }
    auto nval = ndat * ndim * npol * nchan;
    for (uint64_t ival = 0; ival < nval; ival++)
    {
      *data_ptr = (ival % 2 == 0) ? -0.0 : 0.0;
      data_ptr++;
    }
  }

  void GenericVoltageDigitizerTest::setup_random_timeseries_data()
  {
    if (dsp::Operation::verbose)
      std::cerr<< "setup_random_timeseries_data initialising input and output containers" << std::endl;
    input->set_state(Signal::Analytic);
    input->set_order(order);
    input->set_ndim(ndim);
    input->set_npol(npol);
    input->set_nchan(nchan);
    input->set_rate(1.0);
    if (dsp::Operation::verbose)
      std::cerr<< "setup_random_timeseries_data input container initialised" << std::endl;
    output->zero();
    if (dsp::Operation::verbose)
      std::cerr<< "setup_random_timeseries_data output container initialised" << std::endl;

    // ensure we test scaling coming from input
    input->set_scale(1.1);

    if (use_wts)
    {
      input_wts->resize(ndat);
      input_wts->set_ndat_per_weight(ndat_per_weight);
      generate_weights();
    }
    else
    {
      input->resize(ndat);
    }

    float *data_ptr{nullptr};
    switch (order)
    {
    case dsp::TimeSeries::OrderFPT:
      data_ptr = input->get_datptr();
      break;

    case dsp::TimeSeries::OrderTFP:
    default:
      data_ptr = input->get_dattfp();
      break;
    }
    uint64_t num_records = nchan * ndim * npol * ndat;
    generate_random_data(data_ptr, static_cast<size_t>(num_records));

#ifdef HAVE_CUDA
    if(on_gpu)
    {
      if (dsp::Operation::verbose)
      {
        std::cerr << "Transferring input_data from Host to Device" << std::endl;
      }
      TransferCUDATestHelper xfer(device_memory.get(), stream);
      xfer.copy(device_input, input, cudaMemcpyHostToDevice);
    }
#endif
  }

  uint16_t GenericVoltageDigitizerTest::expected_weight(unsigned ichan, unsigned ipol, uint64_t iweight)
  {
    unsigned state = iweight % 3;
    switch (state)
    {
      case 0: return iweight;
      case 1: return ichan;
      case 2: return ipol + 256;
    }
    return 0;
  }


  void GenericVoltageDigitizerTest::generate_weights(int _ndat_per_weight)
  {
    input_wts->set_nchan_weight(input->get_nchan());
    input_wts->set_npol_weight(1);
    input_wts->set_ndat_per_weight(_ndat_per_weight);
    input_wts->resize(ndat);

    unsigned npol_weight = input_wts->get_npol_weight();
    unsigned nchan_weight = input_wts->get_nchan_weight();
    auto nweights = input_wts->get_nweights();

    if (dsp::Operation::verbose)
    {
      std::cerr << "generate_weights ndat: " << ndat << std::endl;
      std::cerr << "generate_weights _ndat_per_weight: " << _ndat_per_weight << std::endl;
      std::cerr << "generate_weights npol_weight: " << npol_weight << std::endl;
      std::cerr << "generate_weights nchan_weight: " << nchan_weight << std::endl;
      std::cerr << "generate_weights ndat_per_weight: " << _ndat_per_weight << std::endl;
      std::cerr << "generate_weights nweights: " << nweights << std::endl;
    }

    for (unsigned ichan=0; ichan<nchan_weight; ichan++)
    {
      for (unsigned ipol=0; ipol<npol_weight; ipol++)
      {
        uint16_t* ptr = input_wts->get_weights(ichan, ipol);
        assert(ptr != nullptr);

        uint64_t ival = 0;
        for (uint64_t iweight=1; iweight<=nweights; iweight++)
        {
          uint16_t weight = expected_weight(ichan, ipol, iweight);
          if (dsp::Operation::verbose)
            std::cerr << "about to set generate_weights ptr[" << ival << "]: " << weight
              << ", ichan=" << ichan << ", ipol=" << ipol << ", iweight=" << iweight << " ptr=" << reinterpret_cast<void *>(ptr) << std::endl;
          ptr[ival] = weight;
          if (dsp::Operation::verbose)
            std::cerr << "generate_weights ptr[" << ival << "]: " << ptr[ival] << std::endl;
          ival++;
        }
      }
    }
  }

  void GenericVoltageDigitizerTest::setup_known_data_timeseries()
  {
    output->zero();
    ndim = 2;
    nchan = 2;
    npol = 2;
    ndat = 4;

    input->set_state(Signal::Analytic);
    input->set_order(order);
    input->set_ndim(ndim);
    input->set_npol(npol);
    input->set_nchan(nchan);
    input->set_rate(1.0);
    input->resize(ndat);

    float *data_ptr{nullptr};
    switch (order)
    {
    case dsp::TimeSeries::OrderFPT:
      data_ptr = input->get_datptr();
      break;

    case dsp::TimeSeries::OrderTFP:
    default:
      data_ptr = input->get_dattfp();
      break;
    }

    if (use_wts)
    {
      generate_weights(2);
    }

    // the first 2 values will clip when using NBIT=8 as the scale used is 10.1
    std::array<float, 32> input_data{
        -12.77228, 12.6336634, -5.005, -1.4842639, -2.5428553,
        0.7414923, -2.1585503, 2.3708253, -0.40148783, -7.3821115,
        1.2647325, -2.0321102, 0.7521466, -0.5648981, 2.9565287,
        4.120244, 0.47453195, -3.2339506, 1.2360413, -1.4815402,
        -1.1341237, -4.9306426, -1.9723871, -3.630312, -2.237967,
        2.8785143, -6.6134176, 1.2712177, -2.2587833, 2.0703325,
        0.95531327, 1.2340846};

    std::copy(input_data.begin(), input_data.end(), data_ptr);
#ifdef HAVE_CUDA
      if(on_gpu)
      {
        if (dsp::Operation::verbose)
        {
          std::cerr << "Transferring input_data from Host to Device" << std::endl;
        }
        TransferCUDATestHelper xfer(device_memory, stream);
        xfer.copy(device_input, input, cudaMemcpyHostToDevice);
      }
#endif
  }

  auto GenericVoltageDigitizerTest::encode_idat_ichan_ipol_idim(uint64_t idat, unsigned ichan, unsigned ipol, unsigned idim) -> float
  {
    float value = (((static_cast<float>(idim + 1) / 10.0) + static_cast<float>(idat + 1)) / 10.0
      + static_cast<float>(ichan + 1)) / 10.0;
    if (ipol == 1)
      value = -value;

    return value;
  }

  void GenericVoltageDigitizerTest::setup_nchanpol_timeseries()
  {
    output->zero();
    input->zero();
    ndat = 4;

    if (dsp::Operation::verbose)
      std::cerr << "setup_nchanpol_timeseries - ndat=" << ndat << ", nchan=" << nchan << ", npol=" << npol << ", ndim=" << ndim << std::endl;

    input->set_state(Signal::Analytic);
    input->set_order(order);
    input->set_ndim(ndim);
    input->set_npol(npol);
    input->set_nchan(nchan);
    input->set_rate(1.0);

    if (use_wts)
    {
      input_wts->zero();
      input_wts->set_ndat_per_weight(2);
      input_wts->set_nchan_weight(nchan);
      input_wts->resize(ndat);
      generate_weights(ndat);
    }
    else
    {
      input->resize(ndat);
    }

    float *data_ptr{nullptr};
    switch (order)
    {
    case dsp::TimeSeries::OrderFPT:
      {
        for (unsigned ichan = 0; ichan < nchan; ichan++)
        {
          for (unsigned ipol = 0; ipol < npol; ipol++)
          {
            data_ptr = input->get_datptr(ichan, ipol);
            for (unsigned idat = 0; idat < ndat; idat++)
            {
              for (unsigned idim = 0; idim < ndim; idim++)
              {
                *data_ptr = encode_idat_ichan_ipol_idim(idat, ichan, ipol, idim);
                data_ptr++;
              }
            }
          }
        }
      }
      break;

    case dsp::TimeSeries::OrderTFP:
    default:
      {
        data_ptr = input->get_dattfp();
        for (unsigned idat = 0; idat < ndat; idat++)
        {
          for (unsigned ichan = 0; ichan < nchan; ichan++)
          {
            for (unsigned ipol = 0; ipol < npol; ipol++)
            {
              for (unsigned idim = 0; idim < ndim; idim++)
              {
                *data_ptr = encode_idat_ichan_ipol_idim(idat, ichan, ipol, idim);
                data_ptr++;
              }
            }
          }
        }
      }
      break;
    }

#ifdef HAVE_CUDA
      if(on_gpu)
      {
        if (dsp::Operation::verbose)
        {
          std::cerr << "Transferring input_data from Host to Device" << std::endl;
        }
        TransferCUDATestHelper xfer(device_memory, stream);
        xfer.copy(device_input, input, cudaMemcpyHostToDevice);
      }
#endif
  }

  auto GenericVoltageDigitizerTest::expected_output_data_tfp() -> const int8_t *
  {
    switch (nbit)
    {
    case 1:
      return tfp_expected_output_nbit1;
    case 2:
      return tfp_expected_output_nbit2;
    case 4:
      return tfp_expected_output_nbit4;
    case 8:
      return tfp_expected_output_nbit8;
    case 16:
      return reinterpret_cast<const int8_t *>(tfp_expected_output_nbit16);
    case -32:
      return reinterpret_cast<const int8_t *>(tfp_expected_output_float);

    default:
      throw std::runtime_error("unknown nbit");
    }
  }

  auto GenericVoltageDigitizerTest::expected_output_data_fpt() -> const int8_t *
  {
    switch (nbit)
    {
    case 1:
      return fpt_expected_output_nbit1;
    case 2:
      return fpt_expected_output_nbit2;
    case 4:
      return fpt_expected_output_nbit4;
    case 8:
      return fpt_expected_output_nbit8;
    case 16:
      return reinterpret_cast<const int8_t *>(fpt_expected_output_nbit16);
    case -32:
      return reinterpret_cast<const int8_t *>(fpt_expected_output_float);
    default:
      throw std::runtime_error("unknown nbit");
    }
  }

  void GenericVoltageDigitizerTest::assert_output_wts()
  {
    if (dsp::Operation::verbose)
      std::cerr << "assert_output_wts" << std::endl;

    const auto nchan_weight = input_wts->get_nchan_weight();
    const auto npol_weight = input_wts->get_npol_weight();
    const auto nweights = input_wts->get_nweights();

    auto outptr = reinterpret_cast<uint16_t *>(output_wts->get_rawptr());

    for (uint64_t iweight = 0; iweight < nweights; iweight++)
    {
      for (unsigned ichan = 0; ichan < nchan_weight; ichan++)
      {
        for (unsigned ipol = 0; ipol < npol_weight; ipol++)
        {
          auto expected_weights_val = expected_weight(ichan, ipol, iweight+1);
          auto actual_weights_val = *outptr;

          ASSERT_EQ(expected_weights_val, actual_weights_val) << " expected_weights_val: "
            << expected_weights_val << " actual_weights_val: " << actual_weights_val;

          outptr ++;
        }
      }
    }
  }

  void GenericVoltageDigitizerTest::assert_known_data()
  {
#ifdef HAVE_CUDA
    if (on_gpu)
    {
      TransferCUDATestHelper xfer;
      xfer.copy_bitseries(output, device_output, cudaMemcpyDeviceToHost);
      if (use_wts)
      {
        xfer.copy_bitseries(output_wts, device_output_wts, cudaMemcpyDeviceToHost);
      }
    }
#endif
    switch (order)
    {
    case dsp::TimeSeries::OrderFPT:
      assert_fpt_known_data();
      break;

    case dsp::TimeSeries::OrderTFP:
    default:
      assert_tfp_known_data();
      break;
    }
  }

  void GenericVoltageDigitizerTest::assert_tfp_known_data()
  {
    // the first 2 values should be clipped
    auto output_data_tfp = expected_output_data_tfp();
    size_t size = input->get_ndat() * abs(nbit) / BITS_PER_BYTE;

    const int8_t *out_ptr = reinterpret_cast<const int8_t *>(output->get_rawptr());

    for (size_t idx = 0; idx < size; idx++)
    {
      const int expected_value = output_data_tfp[idx];
      const int actual_value = static_cast<int>(out_ptr[idx]);
      ASSERT_EQ(actual_value, expected_value) << " idx=" << idx;
    }
  }

  void GenericVoltageDigitizerTest::assert_fpt_known_data()
  {
    // the first 2 values should be clipped
    auto output_data_fpt = expected_output_data_fpt();
    size_t size = input->get_ndat() * abs(nbit) / BITS_PER_BYTE;
    const int8_t *out_ptr = reinterpret_cast<const int8_t *>(output->get_rawptr());

    if (use_wts)
    {
      assert_output_wts();
    }

    for (size_t idx = 0; idx < size; idx++)
    {
      const int expected_value = output_data_fpt[idx];
      const int actual_value = static_cast<int>(out_ptr[idx]);

      ASSERT_EQ(actual_value, expected_value) << " idx=" << idx;
    }
  }

  int GenericVoltageDigitizerTest::calculate_expected_value(float input_value)
  {
    float scale = 1.0;
    float offset = 0.0;
    switch (nbit)
    {
    case 1:
      return !true_signbit_float(input_value);
    case 2:
      scale = 1.03;
      offset = -0.5;
      break;
    case 4:
      scale = 3.14;
      offset = -0.5;
      break;
    case 8:
      scale = 10.1;
      offset = -0.5;
      break;
    case 16:
      scale = 1106.4;
      offset = -0.5;
      break;
    default:
      throw std::runtime_error("invalid nbit");
    }
    int min = -std::pow(2, nbit - 1);
    int max = std::pow(2, nbit - 1) - 1;

    float effective_scale = scale / (input->get_scale());
    const float scaled_val = std::round(fmaf(input_value, effective_scale, offset)) /* * scale + mean */;
    int expected_value = static_cast<int>(scaled_val);

    expected_value = std::max(expected_value, min);
    expected_value = std::min(expected_value, max);

    return expected_value;
  }

  auto GenericVoltageDigitizerTest::unpack_value(unsigned char value, unsigned outidx) -> int
  {
    if (nbit == 8)
      return static_cast<int>(*reinterpret_cast<int8_t *>(&value));

    const unsigned char mask = static_cast<unsigned char>(pow(2, nbit)) - 1;

    auto byte_sample_idx = outidx % (BITS_PER_BYTE / nbit);
    auto bit_shift = byte_sample_idx * nbit;

    unsigned char bitshifted = (unsigned char)(value >> bit_shift);
    int result = (int)(bitshifted & mask);

    if (nbit == 1)
      return result;

    // as we have an arbitrary number of bits we need to do some bit wrangling to get negative values.
    // the most significant bit (MSB) is an indicator of if the number is -ve of not. If the MSB is
    // set need to OR the result with the above masked but bitfliped.
    uint8_t msb = 1 << (nbit - 1);
    if (result & msb)
      result |= ~mask;

    return result;
  }

  void GenericVoltageDigitizerTest::assert_generated_data()
  {
    float *data_ptr;

    switch (order)
    {
      case dsp::TimeSeries::OrderFPT:
        data_ptr = input->get_datptr();
        break;

      case dsp::TimeSeries::OrderTFP:
      default:
        data_ptr = input->get_dattfp();
        break;
    }

#ifdef HAVE_CUDA
    if (on_gpu)
    {
      TransferCUDATestHelper xfer;
      xfer.copy_bitseries(output, device_output, cudaMemcpyDeviceToHost);
      if (use_wts)
      {
        xfer.copy_bitseries(output_wts, device_output_wts, cudaMemcpyDeviceToHost);
      }
    }
#endif

    if (use_wts)
    {
      assert_output_wts();
    }

    int8_t *out_ptr = reinterpret_cast<int8_t *>(output->get_rawptr());
    for (uint64_t idat = 0; idat < ndat; idat++)
    {
      for (unsigned ichan = 0; ichan < nchan; ichan++)
      {
        for (unsigned ipol = 0; ipol < npol; ipol++)
        {
          for (unsigned idim = 0; idim < ndim; idim++)
          {
            uint64_t output_offset = ((idat * nchan + ichan) * npol + ipol) * ndim + idim;
            // Input and output are both TFP
            uint64_t input_offset;
            switch (order)
            {
              case dsp::TimeSeries::OrderFPT:
                input_offset = ((ichan * npol + ipol) * ndat + idat) * ndim + idim;
                break;

              case dsp::TimeSeries::OrderTFP:
              default:
                input_offset = output_offset;
                break;
            }

            const float input_value = data_ptr[input_offset];

            switch (nbit)
            {
            case -32:
            {
              auto actual_value = reinterpret_cast<float *>(out_ptr)[output_offset];
              auto expected_value = input_value / input->get_scale();

              ASSERT_FLOAT_EQ(actual_value, expected_value) << "nbit=" << nbit << " idat="
                                                            << idat << ", ichan=" << ichan << ", ipol" << ipol << ", idim="
                                                            << idim << ", input_offset=" << input_offset << ", output_offset="
                                                            << output_offset << ", input_value=" << input_value;
              break;
            }
            case 16:
            {
              const int expected_value = calculate_expected_value(input_value);
              auto actual_value = reinterpret_cast<int16_t *>(out_ptr)[output_offset];

              ASSERT_EQ(actual_value, expected_value) << "nbit=" << nbit << " idat="
                                                      << idat << ", ichan=" << ichan << ", ipol" << ipol << ", idim="
                                                      << idim << ", input_offset=" << input_offset << ", output_offset="
                                                      << output_offset << ", input_value=" << input_value << ", input_scale="
                                                      << input->get_scale();

              break;
            }
            default:
            {
              const int expected_value = calculate_expected_value(input_value);
              auto bitpacked_output_offset = output_offset / (BITS_PER_BYTE / nbit);

              auto curr_byte = static_cast<unsigned char>(out_ptr[bitpacked_output_offset]);
              auto actual_value = unpack_value(curr_byte, output_offset);

              #ifdef _TRACE
              std::cerr << "nbit=" << nbit << " idat="
                        << idat << ", ichan=" << ichan << ", ipol" << ipol << ", idim="
                        << idim << ", input_offset=" << input_offset << ", output_offset="
                        << output_offset << ", input_value=" << input_value
                        << ", actual_value=" << actual_value << ", expected_value=" << expected_value << std::endl;
              #endif // _TRACE

              ASSERT_EQ(actual_value, expected_value) << "nbit=" << nbit << " idat="
                                                      << idat << ", ichan=" << ichan << ", ipol" << ipol << ", idim="
                                                      << idim << ", input_offset=" << input_offset << ", output_offset="
                                                      << output_offset << ", input_value=" << input_value << ", input_scale="
                                                      << input->get_scale()
                                                      << ", expected_value=" << expected_value
                                                      << ", actual_value=" << actual_value;
            }
            }
          }
        }
      }
    }
  }

  void GenericVoltageDigitizerTest::prepare_transform_input_and_outputs(std::shared_ptr<dsp::GenericVoltageDigitizer> gvd)
  {
    gvd->set_nbit(nbit);
    if (!on_gpu)
    {
      if (use_wts)
      {
        gvd->set_input(input_wts);
        gvd->set_output_weights(output_wts);
      }
      else
      {
        gvd->set_input(input);
      }
      gvd->set_output(output);
    }
#ifdef HAVE_CUDA
    else
    {
      TransferCUDATestHelper xfer;
      xfer.copy(device_input, input, cudaMemcpyHostToDevice);
      gvd->set_input(device_input);
      gvd->set_output(device_output);

      if (use_wts)
      {
        gvd->set_output_weights(device_output_wts);
      }
    }
#endif
  }

  bool GenericVoltageDigitizerTest::perform_transform(std::shared_ptr<dsp::GenericVoltageDigitizer> gvd)
  try
  {
    prepare_transform_input_and_outputs(gvd);
    gvd->prepare();
    gvd->operate();

    assert_metrics(gvd);
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

  TEST_F(GenericVoltageDigitizerTest, test_construct_delete) // NOLINT
  {
    std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());
    ASSERT_NE(gvd, nullptr);
    gvd = nullptr;
    ASSERT_EQ(gvd, nullptr);
  }

  TEST_P(GenericVoltageDigitizerTest, test_digitization_random_data_large) // NOLINT
  {
    // use "inconvenient" non power of two values
    nchan = 33;
    ndat = 128;
    std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());

    ASSERT_NO_THROW(setup_random_timeseries_data());
    ASSERT_TRUE(perform_transform(gvd));
    assert_generated_data();
  }

  TEST_F(GenericVoltageDigitizerTest, test_digitization_nbit1_zeros_tfp) // NOLINT
  {
    nbit = 1;
    order = dsp::TimeSeries::OrderTFP;
    std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());

    ASSERT_NO_THROW(setup_zeroed_data());
    ASSERT_TRUE(perform_transform(gvd));
    assert_generated_data();
  }

  TEST_F(GenericVoltageDigitizerTest, test_digitization_nbit1_zeros_fpt) // NOLINT
  {
    nbit = 1;
    order = dsp::TimeSeries::OrderFPT;
    std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());

    ASSERT_NO_THROW(setup_zeroed_data());
    ASSERT_TRUE(perform_transform(gvd));
    assert_generated_data();
  }

  TEST_P(GenericVoltageDigitizerTest, test_digitization_random_data) // NOLINT
  {
    std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());

    ASSERT_NO_THROW(setup_random_timeseries_data());
    ASSERT_TRUE(perform_transform(gvd));
    assert_generated_data();
  }

  TEST_P(GenericVoltageDigitizerTest, test_digitization_known_data) // NOLINT
  {
    std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());
    ASSERT_NO_THROW(setup_known_data_timeseries());
    ASSERT_TRUE(perform_transform(gvd));

    assert_known_data();
  }

  TEST_P(GenericVoltageDigitizerTest, test_nchanpol_less_than_32) // NOLINT
  {
    for (unsigned ichan = 1; ichan < 32; ichan++)
    {
      for (unsigned ipol = 1; ipol <= 2; ipol++)
      {
        if (ichan * ipol >= 32)
        {
          continue;
        }

        nchan = ichan;
        npol = ipol;
        setup_nchanpol_timeseries();

        std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());

        ASSERT_TRUE(perform_transform(gvd));
        assert_generated_data();
      }
    }
  }

  // ensure that calling prepare correctly prepares the outputs
  TEST_P(GenericVoltageDigitizerTest, test_prepare) // NOLINT
  {
    std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());

    ASSERT_NO_THROW(setup_known_data_timeseries());
    prepare_transform_input_and_outputs(gvd);

    ASSERT_NO_THROW(gvd->prepare());

    Reference::To<dsp::TimeSeries> in = input;
    Reference::To<dsp::BitSeries> out = output;
    Reference::To<dsp::WeightedTimeSeries> in_wts = input_wts;
    Reference::To<dsp::BitSeries> out_wts = output_wts;

    if (on_gpu)
    {
      in = device_input;
      out = device_output;
      in_wts = device_input_wts;
      out_wts = device_output_wts;
    }

    // ensure that nbit=-32 is converted to nbit=32
    unsigned expected_out_nbit = static_cast<unsigned>(std::abs(nbit));

    // validate some simple dimensions to ensure the output has been configured
    ASSERT_EQ(in->get_nchan(), out->get_nchan());
    ASSERT_EQ(in->get_npol(), out->get_npol());
    ASSERT_EQ(in->get_ndim(), out->get_ndim());
    ASSERT_EQ(in->get_state(), out->get_state());
    ASSERT_EQ(in->get_rate(), out->get_rate());
    ASSERT_EQ(out->get_nbit(), expected_out_nbit);

    // additional tests of the weighted timeseries output
    if (use_wts)
    {
      constexpr unsigned expected_out_wts_nbit = 16;
      constexpr unsigned expected_out_wts_ndim = 1;
      double expected_out_wts_rate = in_wts->get_rate() / in_wts->get_ndat_per_weight();

      ASSERT_EQ(in_wts->get_nchan_weight(), out_wts->get_nchan());
      ASSERT_EQ(in_wts->get_npol_weight(), out_wts->get_npol());
      ASSERT_EQ(in_wts->get_state(), out_wts->get_state());
      ASSERT_EQ(out_wts->get_rate(), expected_out_wts_rate);
      ASSERT_EQ(out_wts->get_nbit(), expected_out_wts_nbit);
      ASSERT_EQ(out_wts->get_ndim(), expected_out_wts_ndim);
    }
  }

  TEST_F(GenericVoltageDigitizerTest, test_set_nbit) // NOLINT
  {
    std::shared_ptr<dsp::GenericVoltageDigitizer> gvd(new_device_under_test());

    for (auto nbit = -32; nbit < 100; nbit++)
    {
      switch (nbit)
      {
      case 1:
      case 2:
      case 4:
      case 8:
      case 16:
      case -32:
        ASSERT_NO_THROW(gvd->set_nbit(nbit));
        break;
      default:
        ASSERT_ANY_THROW(gvd->set_nbit(nbit));
        break;
      }
    }
  }

  std::vector<int> nbits{1, 2, 4, 8, 16, -32};

  std::vector<std::tuple<int, dsp::TimeSeries::Order, bool, bool>> get_test_parameters() {
    std::vector<std::tuple<int, dsp::TimeSeries::Order, bool, bool>> params{};
    std::vector<bool> wts_options = {false, true};
    std::vector<dsp::TimeSeries::Order> order_options = {dsp::TimeSeries::OrderTFP, dsp::TimeSeries::OrderFPT};

    for (auto nbit: nbits)
    {
      for (auto on_gpu: get_gpu_flags())
      {
        for (auto use_wts : wts_options)
        {
          for (auto order: order_options)
          {
            params.push_back(std::make_tuple(nbit, order, on_gpu, use_wts));
          }
        }
      }
    }

    return params;
  }

  INSTANTIATE_TEST_SUITE_P(
     GenericVoltageDigitizerTestSuite, GenericVoltageDigitizerTest,
      testing::ValuesIn(get_test_parameters()),
      [](const testing::TestParamInfo<GenericVoltageDigitizerTest::ParamType> &info)
      {
        auto nbit = std::get<0>(info.param);
        auto order = std::get<1>(info.param);
        bool on_gpu = std::get<2>(info.param);
        bool use_wts = std::get<3>(info.param);

        std::string name;

        if (order == dsp::TimeSeries::OrderFPT)
          name += "fpt_";
        else
          name += "tfp_";

        name += + "nbit" + std::to_string(abs(nbit));

        if (on_gpu)
          name += "_on_gpu";
        else
          name += "_on_cpu";

        if (use_wts)
          name += "_use_wts";
        else
          name += "_no_wts";

        return name;
      }); // NOLINT

} // namespace dsp::test
