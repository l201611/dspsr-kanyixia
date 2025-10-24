/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos, Will Gauvin and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ChanPolSelectTest.h"
#include "dsp/TFPOffset.h"
#include "dsp/TimeSeries.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/GtestMain.h"
#include "dsp/Memory.h"
#include "dsp/SignalStateTestHelper.h"

#include <algorithm>
#include <cassert>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_CUDA
#include "dsp/ChanPolSelectCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#include <cuda.h>
#endif

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

ChanPolSelectTest::ChanPolSelectTest()
{
  states = { Signal::Nyquist, Signal::Analytic, Signal::Intensity, Signal::PPQQ, Signal::Coherence };

#ifdef HAVE_CUDA
  device_memory = new CUDA::DeviceMemory;
#endif
}

void ChanPolSelectTest::init_containers()
{
  if (use_wts)
  {
    input = input_wts = new dsp::WeightedTimeSeries;
    output = output_wts = new dsp::WeightedTimeSeries;
  }
  else
  {
    input = new dsp::TimeSeries;
    output = new dsp::TimeSeries;
    input_wts = output_wts = nullptr;
  }

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    if (use_wts)
    {
      device_input = device_input_wts = new dsp::WeightedTimeSeries;
      device_output = device_output_wts = new dsp::WeightedTimeSeries;
    }
    else
    {
      device_input = new dsp::TimeSeries;
      device_output = new dsp::TimeSeries;
      device_input_wts = device_output_wts = nullptr;
    }
    device_input->set_memory(device_memory);
    device_output->set_memory(device_memory);
    if (use_wts)
    {
      device_input_wts->set_weights_memory(device_memory);
      device_output_wts->set_weights_memory(device_memory);
    }
  }
#endif
}

void ChanPolSelectTest::SetUp()
{
}

void ChanPolSelectTest::TearDown()
{
  // ensure the containers are deleted
  input = nullptr;
  output = nullptr;
  device_input = nullptr;
  device_output = nullptr;
}

void ChanPolSelectTest::assert_metrics(dsp::ChanPolSelect* cps)
{
  if (cps)
  {
    auto metrics = cps->get_performance_metrics();

    if (dsp::Operation::verbose)
    {
      std::cerr << "ChanPolSelectTest::assert_metrics" << std::endl;
      std::cerr << "Total processing time: " << metrics->total_processing_time << " seconds" << std::endl;
      std::cerr << "Total time spanned by the data: " << metrics->total_data_time << " seconds"  << std::endl;
      std::cerr << "Total bytes processed: " << metrics->total_bytes_processed << std::endl;
    }
    ASSERT_TRUE(metrics->total_processing_time >= 0); // may need nanosecond precision for tests that are too fast
    ASSERT_TRUE(metrics->total_data_time > 0);
    ASSERT_TRUE(metrics->total_bytes_processed > 0);
  }
  else
  {
    std::cerr << "No ChanPolSelect instance provided for metrics display." << std::endl;
  }
}

int ChanPolSelectTest::generate_random_number(int min, int max)
{
  // Seed the random number generator
  std::uniform_int_distribution<int> distribution(min, max);

  // Generate and return the random number
  return distribution(generator);
}

void ChanPolSelectTest::generate_data()
{
  if (order == dsp::TimeSeries::OrderFPT)
    generate_fpt();
  else
    generate_tfp();
}

float ChanPolSelectTest::get_expected_dat(unsigned ichan, unsigned ipol, uint64_t idat)
{
  // return deterministic number where the each channel, pol and dat should have mostly unique values
  float sign = (ipol % 2) ? 1.0f : -1.0f;
  return sign * static_cast<float>(idat) + static_cast<float>(ichan) / nchan;
}

uint16_t ChanPolSelectTest::get_expected_weight(unsigned ichan, unsigned ipol, uint64_t iweight)
{
  // the 2 most significant bits contain ipol, bottom 14 bits contain iweight
  uint64_t iweight_limit = 1 << 14;
  if (iweight % 2)
    return (static_cast<uint16_t>(ipol) << 14) + static_cast<uint16_t>(iweight % iweight_limit);
  else
    return static_cast<uint16_t>(ichan);
}

void ChanPolSelectTest::generate_wts()
{
  input_wts->set_ndat_per_weight(ndat_per_weight);
  input_wts->resize(ndat);
  const uint64_t input_nweights = input_wts->get_nweights();

  if (dsp::Observation::verbose)
  {
    std::cerr << "input_wts->get_ndat_per_weight(): " << input_wts->get_ndat_per_weight() << std::endl;
    std::cerr << "input_wts->get_npol_weight(): " << input_wts->get_npol_weight() << std::endl;
    std::cerr << "input_wts->get_nchan_weight(): " << input_wts->get_nchan_weight() << std::endl;
    std::cerr << "input_wts->get_nweights(): " << input_nweights << std::endl;
  }

  for (unsigned ichan=0; ichan<input_wts->get_nchan_weight(); ichan++)
  {
    for (unsigned ipol=0; ipol<input_wts->get_npol_weight(); ipol++)
    {
      auto ptr = input_wts->get_weights(ichan, ipol);
      for (uint64_t iweight=0; iweight<input_nweights; iweight++)
      {
        ptr[iweight] = get_expected_weight(ichan, ipol, iweight);
      }
    }
  }
}

void ChanPolSelectTest::generate_fpt()
{
  input->set_order(dsp::TimeSeries::OrderFPT);
  output->set_order(dsp::TimeSeries::OrderFPT);
  if (use_wts)
  {
    if (dsp::Observation::verbose)
    {
      std::cerr << "ChanPolSelectTest::generate_fpt input order: " << input->get_order() << std::endl;
      std::cerr << "ChanPolSelectTest::generate_fpt input_wts order: " << input_wts->get_order() << std::endl;
    }
    generate_wts();
  }
  else
  {
    input->resize(ndat);
  }

  for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
  {
    for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
    {
      auto ptr = input->get_datptr(ichan, ipol);
      assert(ptr != nullptr);

      uint64_t ival = 0;
      for (uint64_t idat=0; idat<input->get_ndat(); idat++)
      {
        const float val = get_expected_dat(ichan, ipol, idat);
        for (unsigned idim=0; idim<input->get_ndim(); idim++)
        {
          ptr[ival] = val;
          ival++;
        }
      }
    }
  }
}

void ChanPolSelectTest::assert_data(bool nchan_weight_equals_nchan, bool npol_weight_equals_npol)
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
    output->zero();
    TransferCUDATestHelper xfer;
    xfer.copy(output, device_output, cudaMemcpyDeviceToHost);
  }
#endif

  if (dsp::Observation::verbose) {
    std::cerr << "ChanPolSelectTest::assert_data input nchan=" << input->get_nchan() << " npol=" << input->get_npol() << std::endl;
    std::cerr << "ChanPolSelectTest::assert_data output nchan=" << output->get_nchan() << " npol=" << output->get_npol() << std::endl;
    std::cerr << "ChanPolSelectTest::assert_data verifying dimensions of output" << std::endl;
  }

  ASSERT_EQ(number_of_channels_to_keep, output->get_nchan());
  ASSERT_EQ(number_of_polarizations_to_keep, output->get_npol());

  ASSERT_EQ(order, output->get_order());

  uint64_t ndat = output->get_ndat();
  ASSERT_EQ(input->get_ndat(), ndat);
  unsigned ndim = output->get_ndim();
  ASSERT_EQ(input->get_ndim(), ndim);

  if (use_wts)
  {
    assert_weights(nchan_weight_equals_nchan, npol_weight_equals_npol);
  }

  if (order == dsp::TimeSeries::OrderFPT)
    assert_fpt();
  else
    assert_tfp();

  if (dsp::Observation::verbose)
   std::cerr << "ChanPolSelectTest::assert_data all values are as expected" << std::endl;
}

void ChanPolSelectTest::assert_weights(bool nchan_weight_equals_nchan, bool npol_weight_equals_npol)
{
  const uint64_t ndat = output_wts->get_ndat();
  const unsigned ndim = output_wts->get_ndim();

  if (dsp::Observation::verbose)
    std::cerr << "ChanPolSelectTest::assert_weights ndat=" << ndat << " ndim=" << ndim << std::endl;

  unsigned error_count = 0;
  unsigned start_chan = 0;
  unsigned start_pol = 0;

  if (nchan_weight_equals_nchan)
  {
    start_chan = start_channel_index;
    ASSERT_EQ(number_of_channels_to_keep, output_wts->get_nchan_weight());
  }
  else
  {
    ASSERT_EQ(1u, output_wts->get_nchan_weight());
  }

  if (npol_weight_equals_npol)
  {
    start_pol = start_polarization_index;
    ASSERT_EQ(number_of_polarizations_to_keep, output_wts->get_npol_weight());
  }
  else
  {
    ASSERT_EQ(1u, output_wts->get_npol_weight());
  }

  const uint64_t output_nweights = output_wts->get_nweights();
  for (unsigned ochan=0; ochan<output_wts->get_nchan_weight(); ochan++)
  {
    unsigned ichan = ochan + start_chan;
    for (unsigned opol=0; opol<output_wts->get_npol_weight(); opol++)
    {
      unsigned ipol = opol + start_pol;
      auto ptr = output_wts->get_weights(ochan, opol);
      ASSERT_NE(ptr, nullptr);
      for (uint64_t iweight=0; iweight<output_nweights; iweight++)
      {
        auto expected = get_expected_weight(ichan, ipol, iweight);
        ASSERT_EQ(expected, ptr[iweight]);
      }
    }
  }

  ASSERT_EQ(error_count, 0u);
}

void ChanPolSelectTest::assert_fpt()
{
  const uint64_t ndat = output->get_ndat();
  const unsigned ndim = output->get_ndim();

  if (dsp::Observation::verbose)
    std::cerr << "ChanPolSelectTest::assert_fpt ndat=" << ndat << " ndim=" << ndim << std::endl;

  unsigned error_count = 0;

  for (unsigned ochan=0; ochan<number_of_channels_to_keep; ochan++)
  {
    unsigned ichan = ochan + start_channel_index;
    for (unsigned opol=0; opol<number_of_polarizations_to_keep; opol++)
    {
      unsigned ipol = opol + start_polarization_index;
      const float* ptr = output->get_datptr(ochan, opol);
      uint64_t ival = 0;

      for (uint64_t idat=0; idat<ndat; idat++)
      {
        float expected = get_expected_dat(ichan, ipol, idat);
        for (unsigned idim=0; idim<ndim; idim++)
        {
          if (ptr[ival] != expected)
          {
            error_count ++;
            std::cerr << "assert_fpt FAIL ichan=" << ichan << " ipol=" << ipol << " idat=" << idat << " idim=" << idim
                      << " val=" << ptr[ival] << " != " << expected << std::endl;
          }

          ival++;
        }
      }
    }
  }

  ASSERT_EQ(error_count, 0u);
}

void ChanPolSelectTest::generate_tfp()
{
  input->set_order(dsp::TimeSeries::OrderTFP);
  output->set_order(dsp::TimeSeries::OrderTFP);

  if (use_wts)
  {
    generate_wts();
  }
  else
  {
    input->resize(ndat);
  }

  dsp::TFPOffset input_offset(input);

  float* ptr = input->get_dattfp();
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
      {
        auto ival = input_offset(idat, ichan, ipol);
        const float val = get_expected_dat(ichan, ipol, idat);
        for (unsigned idim=0; idim<input->get_ndim(); idim++)
        {
          ptr[ival] = val;
          ival ++;
        }
      }
    }
  }
}

void ChanPolSelectTest::assert_tfp()
{
  ASSERT_EQ(dsp::TimeSeries::OrderTFP, output->get_order());

  TFPOffset output_offset(output);

  const uint64_t ndat = output->get_ndat();
  const unsigned ndim = output->get_ndim();

  if (dsp::Observation::verbose)
    std::cerr << "ChanPolSelectTest::assert_tfp ndat=" << ndat << " ndim=" << ndim << std::endl;

  unsigned error_count = 0;

  const float* ptr = output->get_dattfp();

  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned ochan=0; ochan<number_of_channels_to_keep; ochan++)
    {
      unsigned ichan = ochan + start_channel_index;
      for (unsigned opol=0; opol<number_of_polarizations_to_keep; opol++)
      {
        unsigned ipol = opol + start_polarization_index;
        float expected = get_expected_dat(ichan, ipol, idat);

        auto ival = output_offset(idat, ochan, opol);

        for (unsigned idim=0; idim<ndim; idim++)
        {
          if (ptr[ival] != expected)
          {
            error_count ++;
            std::cerr << "assert_tfp FAIL idat=" << idat << " ichan=" << ichan << " ipol=" << ipol << " idim=" << idim
                      << " val=" << ptr[ival] << " != " << expected << std::endl;
          }

          ival++;
        }
      }
    }
  }

  ASSERT_EQ(error_count, 0u);
}

void ChanPolSelectTest::assert_transform_configurations(dsp::ChanPolSelect* cps)
{
  ASSERT_EQ(start_channel_index, cps->get_start_channel_index());
  ASSERT_EQ(number_of_channels_to_keep, cps->get_number_of_channels_to_keep());
  ASSERT_EQ(start_polarization_index, cps->get_start_polarization_index());
  ASSERT_EQ(number_of_polarizations_to_keep, cps->get_number_of_polarizations_to_keep());
}

void ChanPolSelectTest::test_driver(bool random_start_index, bool nchan_weight_equals_nchan, bool npol_weight_equals_npol)
{
  if (dsp::Operation::verbose)
    std::cerr << "ChanPolSelectTest::test_driver"
              << " random_start_index=" << random_start_index
              << " nchan_weight_equals_nchan=" << nchan_weight_equals_nchan
              << " npol_weight_equals_npol=" << npol_weight_equals_npol
              << std::endl;

  for (auto & state : states)
  {
    // iterate through the possible number of polarisations for the polarisation state
    std::vector<unsigned> npols = get_npols_for_state(state);
    for (auto & npol : npols)
    {
      unsigned ndim = get_ndim_for_state(state);

      if (dsp::Observation::verbose)
        std::cerr << "ChanPolSelectTest::test_driver"
                  << " state=" << state_string(state)
                  << " npol=" << npol
                  << std::endl;

      input->set_state(state);
      input->set_npol(npol);
      input->set_ndim(ndim);
      input->set_nchan(nchan);
      input->set_rate(1.0);

      if (use_wts)
      {
        if (nchan_weight_equals_nchan)
        {
          input_wts->set_nchan_weight(input->get_nchan());
        }
        if (npol_weight_equals_npol)
        {
          input_wts->set_npol_weight(input->get_npol());
        }
      }
      if (dsp::Observation::verbose)
        std::cerr << "ChanPolSelectTest::test_driver generate_data" << std::endl;
      generate_data();

      for (unsigned nchan_keep=1;nchan_keep<nchan;nchan_keep++)
      {
        number_of_channels_to_keep = nchan_keep;
        number_of_polarizations_to_keep = 1;
        start_channel_index = 0;
        start_polarization_index = 0;
        if (random_start_index)
        {
          start_channel_index = generate_random_number(0,number_of_channels_to_keep-1);
        }

        Reference::To<dsp::ChanPolSelect> cps(new_device_under_test());
        if (dsp::Observation::verbose)
          std::cerr << "ChanPolSelectTest::test_driver configure dsp::ChanPolSelect" << std::endl;
        EXPECT_NO_THROW(cps->set_number_of_channels_to_keep(number_of_channels_to_keep));
        EXPECT_NO_THROW(cps->set_start_channel_index(start_channel_index));
        EXPECT_NO_THROW(cps->set_number_of_polarizations_to_keep(number_of_polarizations_to_keep));
        assert_transform_configurations(cps);

        if (dsp::Observation::verbose)
          std::cerr << "ChanPolSelectTest::test_driver perform dsp::ChanPolSelect transform" << std::endl;

        if (random_start_index)
        {
          if(number_of_channels_to_keep+start_channel_index<=nchan)
          {
            ASSERT_TRUE(perform_transform(cps, false));
            assert_data(nchan_weight_equals_nchan, npol_weight_equals_npol);
          }
          else
          {
            ASSERT_FALSE(perform_transform(cps, false));
          }
        }
        else
        {
          ASSERT_TRUE(perform_transform(cps, true));
          if (dsp::Observation::verbose)
            std::cerr << "ChanPolSelectTest::test_driver verify dsp::ChanPolSelect output" << std::endl;
          assert_data(nchan_weight_equals_nchan, npol_weight_equals_npol);
        }

        if (dsp::Observation::verbose)
          std::cerr << "ChanPolSelectTest::test_driver destroy dsp::ChanPolSelect" << std::endl;
        cps = nullptr;
      }
    }
  }
}

bool ChanPolSelectTest::perform_transform(dsp::ChanPolSelect* cps, bool print_exceptions)
try
{
  cps->set_input(input);
  cps->set_output(output);

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    TransferCUDATestHelper xfer;
    xfer.copy(device_input, input, cudaMemcpyHostToDevice);
    cps->set_input(device_input);
    cps->set_output(device_output);
  }
#endif

  cps->prepare();
  cps->operate();
  assert_metrics(cps);
  return true;
}
catch (std::exception &exc)
{
  if (print_exceptions)
  {
    std::cerr << "Exception Caught: " << exc.what() << std::endl;
  }
  return false;
}
catch (Error &error)
{
  if (print_exceptions)
  {
    std::cerr << "Error Caught: " << error << std::endl;
  }
  return false;
}

dsp::ChanPolSelect* ChanPolSelectTest::new_device_under_test()
{
  Reference::To<dsp::ChanPolSelect> device = new dsp::ChanPolSelect;
#ifdef HAVE_CUDA
  if (on_gpu)
  {
    device->set_engine(new CUDA::ChanPolSelectEngine);
  }
#endif
  return device.release();
}

TEST_P(ChanPolSelectTest, test_construct_delete) // NOLINT
{
  Reference::To<dsp::ChanPolSelect> csp(new_device_under_test());
  ASSERT_NE(csp, nullptr);
  csp = nullptr;
  ASSERT_FALSE(csp);
}

TEST_P(ChanPolSelectTest, test_nchan_index_zero) // NOLINT
{
  auto param = GetParam();
  on_gpu = std::get<0>(param);
  order = std::get<1>(param);
  use_wts = std::get<2>(param);

  init_containers();
  test_driver(false, true, true);
  if (dsp::Observation::verbose)
    std::cerr << "ChanPolSelectTest::test_nchan_index_zero exit" << std::endl;
}

TEST_P(ChanPolSelectTest, test_nchan_weight_equals_nchan) // NOLINT
{
  auto param = GetParam();
  on_gpu = std::get<0>(param);
  order = std::get<1>(param);
  use_wts = std::get<2>(param);

  init_containers();
  test_driver(false, false, false);
  if (dsp::Observation::verbose)
    std::cerr << "ChanPolSelectTest::test_nchan_weight_equals_nchan exit" << std::endl;
}

TEST_P(ChanPolSelectTest, test_npol_weight_equals_npol) // NOLINT
{
  auto param = GetParam();
  on_gpu = std::get<0>(param);
  order = std::get<1>(param);
  use_wts = std::get<2>(param);

  init_containers();
  test_driver(false, true, false);
  if (dsp::Observation::verbose)
    std::cerr << "ChanPolSelectTest::test_npol_weight_equals_npol exit" << std::endl;
}

TEST_P(ChanPolSelectTest, test_nchan_index_random) // NOLINT
{
  auto param = GetParam();
  on_gpu = std::get<0>(param);
  order = std::get<1>(param);
  use_wts = std::get<2>(param);

  init_containers();
  test_driver(true, true, true);
}

TEST_P(ChanPolSelectTest, test_nchan_induce_error) // NOLINT
{
  auto param = GetParam();
  on_gpu = std::get<0>(param);
  order = std::get<1>(param);
  use_wts = std::get<2>(param);

  init_containers();

  // iterate through the possible polarisation states
  for (auto & state : states)
  {
    // iterate through the possible number of polarisations for the polarisation state
    std::vector<unsigned> npols = get_npols_for_state(state);
    for (auto & npol : npols)
    {
      unsigned ndim = get_ndim_for_state(state);

      if (dsp::Observation::verbose)
        std::cerr << "ChanPolSelectTest::test_nchan_induce_error"
        << " state=" << state_string(state)
        << " npol=" << npol
        << " ndim=" << ndim
        << std::endl;

      input->set_state(state);
      input->set_npol(npol);
      input->set_ndim(ndim);
      input->set_nchan(nchan);

      Reference::To<dsp::ChanPolSelect> cps (new_device_under_test());
      generate_data();

      // test improper values for channels to keep
      number_of_channels_to_keep = 0;
      EXPECT_NO_THROW(cps->set_number_of_channels_to_keep(number_of_channels_to_keep));
      ASSERT_FALSE(perform_transform(cps, false));
      number_of_channels_to_keep = nchan + 1;
      EXPECT_NO_THROW(cps->set_number_of_channels_to_keep(number_of_channels_to_keep));
      ASSERT_FALSE(perform_transform(cps, false));
      // restore to a sensible value
      number_of_channels_to_keep = nchan;
      cps->set_number_of_channels_to_keep(number_of_channels_to_keep);

      // test improper values for start channel index
      start_channel_index = 1;
      EXPECT_NO_THROW(cps->set_start_channel_index(start_channel_index));
      ASSERT_FALSE(perform_transform(cps, false));
      // restore to a sensible value
      start_channel_index = 0;
      cps->set_start_channel_index(start_channel_index);

      // test improper values for polarisations to keep
      number_of_polarizations_to_keep = 0;
      EXPECT_NO_THROW(cps->set_number_of_polarizations_to_keep(number_of_polarizations_to_keep));
      ASSERT_FALSE(perform_transform(cps, false));
      number_of_polarizations_to_keep = 5;
      EXPECT_NO_THROW(cps->set_number_of_polarizations_to_keep(number_of_polarizations_to_keep));
      ASSERT_FALSE(perform_transform(cps, false));
      // restore to a sensible value
      number_of_polarizations_to_keep = npol;
      cps->set_number_of_polarizations_to_keep(number_of_polarizations_to_keep);

      // test improper values for start polarization index
      start_polarization_index = 1;
      EXPECT_NO_THROW(cps->set_start_polarization_index(start_polarization_index));
      ASSERT_FALSE(perform_transform(cps, false));
      // restore to a sensible value
      cps->set_start_polarization_index(0);

      cps = nullptr;
    }
  }
}

std::vector<std::tuple<bool, dsp::TimeSeries::Order, bool>> get_test_parameters()
{
  std::vector<std::tuple<bool, dsp::TimeSeries::Order, bool>> params{};
  std::vector<bool> wts_options = {false, true};
  std::vector<dsp::TimeSeries::Order> order_options = {dsp::TimeSeries::OrderTFP, dsp::TimeSeries::OrderFPT};

  for (auto on_gpu: get_gpu_flags())
  {
    for (auto use_wts: wts_options)
    {
      for (auto order: order_options)
      {
        params.push_back(std::make_tuple(on_gpu, order, use_wts));
      }
    }
  }

  return params;
}

INSTANTIATE_TEST_SUITE_P(ChanPolSelectTestSuite, ChanPolSelectTest,
  testing::ValuesIn(get_test_parameters()),
  [](const testing::TestParamInfo<ChanPolSelectTest::ParamType>& info)
  {
    bool on_gpu = std::get<0>(info.param);
    auto order = std::get<1>(info.param);
    bool use_wts = std::get<2>(info.param);
    std::string name;
    if (on_gpu)
      name = "on_gpu";
    else
      name = "on_cpu";

    if (order == dsp::TimeSeries::OrderFPT)
      name += "_fpt";
    else
      name += "_tfp";

    if (use_wts)
      name += "_wts";
    return name;
  }
); // NOLINT

} // namespace dsp::test
