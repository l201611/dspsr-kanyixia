/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/WeightedTimeSeriesTest.h"
#include "dsp/GtestMain.h"
#include "dsp/Memory.h"

#if HAVE_CUDA
#include <cuda.h>
#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#endif

//! main method passed to googletest
int main(int argc, char *argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test
{
  WeightedTimeSeriesTest::WeightedTimeSeriesTest()
  {
    std::cerr << "WeightedTimeSeriesTest ctor" << std::endl;
  }

  void WeightedTimeSeriesTest::SetUp()
  {
    if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
    {
      info = GetParam();
    }

    sut_host = new dsp::WeightedTimeSeries;

    sut_host->set_ndim(info.ndim);
    sut_host->set_npol(info.npol);
    sut_host->set_nchan(info.nchan);

    sut_host->set_ndat_per_weight(info.ndat_per_weight);
    sut_host->set_nchan_weight(info.nchan_weight);
    sut_host->set_npol_weight(info.npol_weight);
    sut_host->resize(info.ndat);

    current_sut = sut_host;

#ifdef HAVE_CUDA
    if (info.on_gpu)
    {
      if (dsp::Observation::verbose)
      {
        std::cerr << "WeightedTimeSeriesTest::Setup WeightedTimeSeries on GPU" << std::endl;
      }
      cudaError_t result = cudaStreamCreate(&stream);
      ASSERT_EQ(result, cudaSuccess);
      device_memory = new CUDA::DeviceMemory(stream);
      sut_device = new dsp::WeightedTimeSeries;

      sut_device->set_memory(device_memory);
      sut_device->set_weights_memory(device_memory);

      current_sut = sut_device;
    }
#endif
  }

  void WeightedTimeSeriesTest::TearDown()
  {
    sut_host = nullptr;
    sut_device = nullptr;
    device_memory = nullptr;

#ifdef HAVE_CUDA
    if (info.on_gpu)
    {
      cudaError_t result = cudaStreamSynchronize(stream);
      ASSERT_EQ(result, cudaSuccess);
      result = cudaStreamDestroy(stream);
      if (result != cudaSuccess)
        std::cerr << "cudaStreamDestroy: " << cudaGetErrorString(result) << std::endl;
      ASSERT_EQ(result, cudaSuccess);
    }
#endif
  }


  uint16_t WeightedTimeSeriesTest::expected_weight(unsigned ichan, unsigned ipol, uint64_t iweight)
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

  void WeightedTimeSeriesTest::generate_weights()
  {
    unsigned npol_weight = sut_host->get_npol_weight();
    unsigned nchan_weight = sut_host->get_nchan_weight();
    auto nweights = sut_host->get_nweights();

    if (dsp::Observation::verbose)
    {
      std::cerr << "generate_weights ndat: " << info.ndat << std::endl;
      std::cerr << "generate_weights _ndat_per_weight: " << info.ndat_per_weight << std::endl;
      std::cerr << "generate_weights npol_weight: " << npol_weight << std::endl;
      std::cerr << "generate_weights nchan_weight: " << nchan_weight << std::endl;
      std::cerr << "generate_weights nweights: " << nweights << std::endl;
    }

    for (unsigned ichan=0; ichan<nchan_weight; ichan++)
    {
      for (unsigned ipol=0; ipol<npol_weight; ipol++)
      {
        uint16_t* ptr = sut_host->get_weights(ichan, ipol);
        assert(ptr != nullptr);

        for (uint64_t iweight=0; iweight<nweights; iweight++)
        {
          uint16_t weight = expected_weight(ichan, ipol, iweight);
          ptr[iweight] = weight;
        }
      }
    }
  }

  auto WeightedTimeSeriesTest::expected_data(uint64_t idat, unsigned ichan, unsigned ipol, unsigned idim) -> float
  {
    float value = (((static_cast<float>(idim + 1) / 10.0) + static_cast<float>(idat + 1)) / 10.0
      + static_cast<float>(ichan + 1)) / 10.0;
    if (ipol == 1)
      value = -value;

    return value;
  }

  void WeightedTimeSeriesTest::generate_data()
  {
    sut_host->zero();

    float* data_ptr = nullptr;
    switch (info.order)
    {
    case dsp::TimeSeries::OrderFPT:
      {
        for (auto ichan = 0; ichan < info.nchan; ichan++)
        {
          for (auto ipol = 0; ipol < info.npol; ipol++)
          {
            data_ptr = sut_host->get_datptr(ichan, ipol);
            for (auto idat = 0; idat < info.ndat; idat++)
            {
              for (auto idim = 0; idim < info.ndim; idim++)
              {
                *data_ptr = expected_data(idat, ichan, ipol, idim);
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
        data_ptr = sut_host->get_dattfp();
        for (auto idat = 0; idat < info.ndat; idat++)
        {
          for (auto ichan = 0; ichan < info.nchan; ichan++)
          {
            for (auto ipol = 0; ipol < info.npol; ipol++)
            {
              for (auto idim = 0; idim < info.ndim; idim++)
              {
                *data_ptr = expected_data(idat, ichan, ipol, idim);
                data_ptr++;
              }
            }
          }
        }
      }
      break;
    }

    generate_weights();

#ifdef HAVE_CUDA
    if(info.on_gpu)
    {
      if (dsp::Observation::verbose)
      {
        std::cerr << "Transferring input_data from Host to Device" << std::endl;
      }
      TransferCUDATestHelper xfer(device_memory, stream);
      xfer.copy(sut_device, sut_host, cudaMemcpyHostToDevice);
    }
#endif
  }

  void WeightedTimeSeriesTest::assert_weights(const TestInfo& expect, const WeightedTimeSeries* sut)
  {
    if (!sut)
      sut = sut_host;

    if (dsp::Observation::verbose)
      std::cerr << "assert_weights" << std::endl;

    ASSERT_EQ(expect.nchan_weight, sut->get_nchan_weight());
    ASSERT_EQ(expect.npol_weight, sut->get_npol_weight());
    const auto nweights = sut->get_nweights();

    for (unsigned ichan=0; ichan<expect.nchan_weight; ichan++)
    {
      for (unsigned ipol=0; ipol<expect.npol_weight; ipol++)
      {
        auto ptr = sut->get_weights(ichan, ipol);
        assert(ptr != nullptr);

        for (uint64_t iweight=0; iweight<nweights; iweight++)
        {
          auto expected_weights_val = expected_weight(ichan, ipol, iweight);
          auto actual_weights_val = ptr[iweight];

          ASSERT_EQ(expected_weights_val, actual_weights_val) << " expected_weights_val: "
            << expected_weights_val << " actual_weights_val: " << actual_weights_val;
        }
      }
    }
  }

  void WeightedTimeSeriesTest::assert_data(const TestInfo& expect)
  {
#ifdef HAVE_CUDA
    if(info.on_gpu)
    {
      if (dsp::Observation::verbose)
      {
        std::cerr << "Transferring input_data from Device to Host" << std::endl;
      }
      TransferCUDATestHelper xfer(device_memory, stream);
      xfer.copy(sut_host, sut_device, cudaMemcpyDeviceToHost);
    }
#endif

    float* data_ptr = nullptr;
    switch (expect.order)
    {
    case dsp::TimeSeries::OrderFPT:
      {
        for (auto ichan = 0; ichan < expect.nchan; ichan++)
        {
          for (auto ipol = 0; ipol < expect.npol; ipol++)
          {
            data_ptr = sut_host->get_datptr(ichan, ipol);
            for (auto idat = 0; idat < expect.ndat; idat++)
            {
              for (auto idim = 0; idim < expect.ndim; idim++)
              {
                auto expected_value = expected_data(idat, ichan, ipol, idim);
                auto actual_value = *data_ptr;

                ASSERT_EQ(expected_value, actual_value) << " expected_val: "
                  << expected_value << " actual_val: " << actual_value;
                  
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
        data_ptr = sut_host->get_dattfp();
        for (auto idat = 0; idat < expect.ndat; idat++)
        {
          for (auto ichan = 0; ichan < expect.nchan; ichan++)
          {
            for (auto ipol = 0; ipol < expect.npol; ipol++)
            {
              for (auto idim = 0; idim < expect.ndim; idim++)
              {
                auto expected_value = expected_data(idat, ichan, ipol, idim);
                auto actual_value = *data_ptr;

                ASSERT_EQ(expected_value, actual_value) << " expected_val: "
                  << expected_value << " actual_val: " << actual_value;
                  
                data_ptr++;
              }
            }
          }
        }
      }
      break;
    }

    assert_weights(expect);
  }

  /* This test confirms that the generate and assert methods are working as required. */
  TEST_P(WeightedTimeSeriesTest, test_assertions) // NOLINT
  {
    generate_data();
    assert_data(info);
  }

  /* This test confirms that the stride between blocks is not modified
     and that data and weights remain intact after the number of samples is reduced. */
  TEST_P(WeightedTimeSeriesTest, test_downsize) // NOLINT
  {
    generate_data();

    auto data_stride = current_sut->get_stride ();
    auto weights_stride = current_sut->get_weights_stride ();

    TestInfo expect = info;
    expect.ndat = info.ndat / 2;
    current_sut->resize(expect.ndat);

    ASSERT_EQ(data_stride, current_sut->get_stride());
    ASSERT_EQ(weights_stride, current_sut->get_weights_stride());
    assert_data(expect);
  }

  /* This test confirms that the stride between blocks is reduced when ndat is reduced
     but the number of channels is increased. */
  TEST_P(WeightedTimeSeriesTest, test_reshape) // NOLINT
  {
    generate_data();

    auto data_stride = current_sut->get_stride ();
    auto weights_stride = current_sut->get_weights_stride ();

    current_sut->set_nchan(info.nchan * 4);
    current_sut->set_nchan_weight(info.nchan_weight * 4);
    current_sut->resize(info.ndat / 4);

    ASSERT_GT(data_stride, current_sut->get_stride());
    ASSERT_GT(weights_stride, current_sut->get_weights_stride());

    /* If the stride between blocks is *not* reduced as expected, then
       WeightedTimeSeries::get_weight(ichan,ipol) will return pointers to
       out-of-bounds memory for some valid values of ichan and ipol.
       In this case, the following call to generate_data would cause a segfault.
    */
    generate_data();
  }

  /* This test directly calls copy_configuration, testing all four possible cases of host/device copy. */
  TEST_P(WeightedTimeSeriesTest, test_copy_configuration) // NOLINT
  {
    generate_weights();

    Reference::To<dsp::WeightedTimeSeries> copy_host = new dsp::WeightedTimeSeries;

#ifdef HAVE_CUDA

    if (info.on_gpu)
    {
      // host-to-device
      sut_device->copy_configuration(sut_host);

      Reference::To<dsp::WeightedTimeSeries> copy_device = new dsp::WeightedTimeSeries;
      copy_device->set_memory(device_memory);
      copy_device->set_weights_memory(device_memory);

      // device-to-device
      copy_device->copy_configuration(sut_device);

      // device-to-host
      copy_host->copy_configuration(copy_device);
    }

#endif

    if (!info.on_gpu)
    {
      // host-to-host
      copy_host->copy_configuration(sut_host);
    }

    // Regenerating should cause a memory fault if the stride between blocks is not reduced as required.
    assert_weights(info, copy_host);
  }

  std::vector<TestInfo> get_test_parameters()
  {
    std::vector<TestInfo> params;

    TestInfo test;

    for (auto ndat: {11, 1024})
    {
      test.ndat = ndat;
      for (auto nchan: {3, 64})
      {
        test.nchan = nchan;
        for (auto npol: {2, 4})
        {        
          test.npol = npol;
          for (auto ndat_per_weight: {4, 32})
          {
            test.ndat_per_weight = ndat_per_weight;
            for (auto nchan_weight: {1, nchan})
            {
              test.nchan_weight = nchan_weight;
              for (auto npol_weight: {1, npol})
              {
                test.npol_weight = npol_weight;
                for (auto on_gpu: get_gpu_flags())
                {
                  test.on_gpu = on_gpu;
                  params.push_back(test);
                }
              }
            }
          }
        }
      }
    }

    std::cerr << "get_test_parameters returning " << params.size() << " tests" << std::endl;
    return params;
  }

  INSTANTIATE_TEST_SUITE_P(
     WeightedTimeSeriesTestSuite, WeightedTimeSeriesTest,
      testing::ValuesIn(get_test_parameters()),
      [](const testing::TestParamInfo<WeightedTimeSeriesTest::ParamType> &test)
      {
        auto info = test.param;

        std::string name;

        name += "_ndat" + std::to_string(info.ndat);
        name += "_nchan" + std::to_string(info.nchan);
        name += "_npol" + std::to_string(info.npol);
        name += "_ndat_wt" + std::to_string(info.ndat_per_weight);
        name += "_nchan_wt" + std::to_string(info.nchan_weight);
        name += "_npol_wt" + std::to_string(info.npol_weight);

        if (info.on_gpu)
          name += "_on_gpu";
        else
          name += "_on_cpu";

        return name;
      }); // NOLINT

} // namespace dsp::test
