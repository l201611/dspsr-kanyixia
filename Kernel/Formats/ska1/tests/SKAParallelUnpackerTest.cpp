/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dspsr_srcdir.h"

#include "dsp/SKAParallelUnpackerTest.h"
#include "dsp/ASCIIObservation.h"
#include "dsp/GtestMain.h"
#include "dsp/Memory.h"
#include "dsp/TFPOffset.h"
#include "dsp/WeightedTimeSeries.h"
#include "FilePtr.h"

#ifdef HAVE_CUDA
#include "dsp/SKAParallelUnpackerCUDA.h"
#include <cuda.h>
#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib> // For std::getenv

//! main method passed to googletest
int main(int argc, char *argv[])
{
  dsp::test::test_data_dir() = DSPSR_SRCDIR "/Kernel/Formats/ska1/tests/data/";

  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

SKAParallelUnpackerTest::SKAParallelUnpackerTest()
{
}

void SKAParallelUnpackerTest::SetUp() try
{
  auto param = GetParam();
  on_gpu = param.on_gpu;
  cbf_name = param.cbf_name;

  input = new dsp::ParallelBitSeries;
  output = new dsp::WeightedTimeSeries;

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    cudaError_t result = cudaStreamCreate(&stream);
    ASSERT_EQ(result, cudaSuccess);
    device_input = new dsp::ParallelBitSeries;
    device_output = new dsp::WeightedTimeSeries;
    device_memory = new CUDA::DeviceMemory;
    device_input->resize(nbitseries);
    device_input->set_memory(device_memory);
    device_output->set_memory(device_memory);
    device_output->set_weights_memory(device_memory);
    device_output->set_order(order);
  }
#endif
  generate_data();
}
catch(Error& error)
{
  std::cerr << "SKAParallelUnpackerTest::SetUp exception " << error << std::endl;
  throw error;
}

void SKAParallelUnpackerTest::TearDown()
{
  obs_data=nullptr;
  obs_weights=nullptr;
  input=nullptr;
  output=nullptr;

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    cudaError_t result = cudaStreamSynchronize(stream);
    ASSERT_EQ(result, cudaSuccess);
    result = cudaStreamDestroy(stream);
    ASSERT_EQ(result, cudaSuccess);
  }

  device_input = nullptr;
  device_output = nullptr;
  device_memory = nullptr;
  device_input = nullptr;
  device_output = nullptr;
#endif

}

std::string SKAParallelUnpackerTest::load_config_from_file(const std::string& filename)
{
  FilePtr fptr = fopen (filename.c_str(), "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::DADAFile::get_header",
		 "fopen (%s)", filename.c_str());

  // default DADA header size
  unsigned hdr_size = 16384;
  std::vector<char> buffer;
  char* header = 0;

  ::rewind (fptr);

  buffer.resize (hdr_size);
  header = &(buffer[0]);

  size_t nbytes =  fread (header, 1, hdr_size, fptr);

  // ensure that text is null-terminated before calling ascii_header_get
  header[ nbytes-1 ] = '\0';

  return std::string(header);

}

uint32_t SKAParallelUnpackerTest::get_expected_scale(uint32_t ipacket)
{
  return ipacket + 1;
}

int16_t SKAParallelUnpackerTest::get_expected_data_16b(uint32_t ival, uint32_t ipol, uint32_t scale)
{
  int32_t scaled_value = static_cast<int32_t>(ival * scale);
  if (ipol == 0)
    scaled_value *= -1;

  scaled_value = std::max(-32768, scaled_value);
  scaled_value = std::min(32767, scaled_value);
  return static_cast<int16_t>(scaled_value);
}

int8_t SKAParallelUnpackerTest::get_expected_data_8b(uint32_t ival, uint32_t ipol, uint32_t scale)
{
  int32_t scaled_value = static_cast<int32_t>(ival * scale);
  if (ipol == 0)
    scaled_value *= -1;

  scaled_value = std::max(-128, scaled_value);
  scaled_value = std::min(127, scaled_value);
  return static_cast<int8_t>(scaled_value);
}

uint16_t SKAParallelUnpackerTest::get_expected_weight(uint32_t ichan)
{
  uint32_t scaled_value = std::min(static_cast<uint32_t>(65535), ichan);
  return static_cast<uint16_t>(scaled_value);
}

void SKAParallelUnpackerTest::generate_data()
{
  std::string data_header_filename = cbf_name + "_data_header.txt";
  std::string weights_header_filename = cbf_name + "_weights_header.txt";

  std::string header_data = load_config_from_file(test_data_file(data_header_filename));
  std::string header_weights = load_config_from_file(test_data_file(weights_header_filename));

  obs_data = new dsp::ASCIIObservation(header_data.c_str());
  obs_weights = new dsp::ASCIIObservation(header_weights.c_str());

  obs_data->custom_header_get("UDP_NCHAN", "%u", &nchan_per_packet);
  obs_data->custom_header_get("UDP_NSAMP", "%u", &nsamp_per_packet);

  npol = obs_data->get_npol();
  ndim = obs_data->get_ndim();
  nchan = obs_data->get_nchan();
  nbit = obs_data->get_nbit();

  if (nbit != 8 && nbit != 16)
  {
    std::cerr << "SKAParallelUnpackerTest::generate_data nbit=" << nbit << " but only 8 or 16 is supported" << std::endl;
  }

  unsigned data_size = nchan * npol * ndim * ndat * nbit;
  unsigned weights_size = nchan * obs_weights->get_npol() * obs_weights->get_ndim() * ndat * nbit;

  obs_data->set_ndat(ndat);
  obs_weights->set_ndat(ndat/32);

  input->resize(nbitseries);
  input->set_ndat(ndat);

  input->copy_configuration(obs_data);
  input->at(0)->copy_configuration(obs_data);
  input->at(1)->copy_configuration(obs_weights);

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    device_input->set_ndat(ndat);
    device_input->copy_configuration(obs_data);
    device_input->at(0)->copy_configuration(obs_data);
    device_input->at(1)->copy_configuration(obs_weights);
  }
#endif

  BitSeries* data_bitseries = input->at(0);
  data_bitseries->resize(ndat);

  uint32_t nvalue = data_bitseries->get_npol() * data_bitseries->get_ndim() * data_bitseries->get_nchan() * data_bitseries->get_ndat();
  int16_t* data_ptr_16b = reinterpret_cast<int16_t*>(data_bitseries->get_rawptr());
  int8_t* data_ptr_8b = reinterpret_cast<int8_t*>(data_bitseries->get_rawptr());

  const uint32_t npackets_per_heap = nchan / nchan_per_packet;
  const uint32_t nval_per_packet = nsamp_per_packet * ndim;
  const uint32_t nheaps = ndat / nsamp_per_packet;
  const uint32_t nsamp = nheaps * nsamp_per_packet;

  uint64_t idx=0;

  // data
  for (uint32_t iheap=0; iheap<nheaps; iheap++)
  {
    for (uint32_t ipacket=0; ipacket<npackets_per_heap; ipacket++)
    {
      const uint32_t scale = get_expected_scale(ipacket);
      for (uint32_t ichan=0; ichan<nchan_per_packet; ichan++)
      {
        for (uint32_t ipol=0; ipol<npol; ipol++)
        {
          for (uint32_t ival=0; ival<nval_per_packet; ival++)
          {
            if (nbit == 8)
              data_ptr_8b[idx] = get_expected_data_8b(ival, ipol, scale);
            else
              data_ptr_16b[idx] = get_expected_data_16b(ival, ipol, scale);
            idx++;
          }
        }
      }
    }
  }

  // scales and weights
  BitSeries* weights_bitseries = input->at(1);
  weights_bitseries->resize(ndat);
  unsigned char* base_ptr = weights_bitseries->get_rawptr();

  // the number of bytes for weights in each packet
  const size_t weights_stride = nchan_per_packet * sizeof(uint16_t);

  for (uint32_t iheap=0; iheap<nheaps; iheap++)
  {
    for (uint32_t ipacket=0; ipacket<npackets_per_heap; ipacket++)
    {
      // write the 32-bit float scale factor
      auto scale_ptr = reinterpret_cast<float*>(base_ptr);
      *scale_ptr = static_cast<float>(get_expected_scale(ipacket));
      base_ptr += sizeof(float);

      auto weights_ptr = reinterpret_cast<uint16_t*>(base_ptr);
      base_ptr += weights_stride;
      for (uint32_t ichan=0; ichan<nchan_per_packet; ichan++)
      {
        weights_ptr[ichan] = get_expected_weight(ipacket * nchan_per_packet + ichan);
      }
    }
  }
}

void SKAParallelUnpackerTest::assert_output()
{
#if HAVE_CUDA
  if (on_gpu)
  {
    TransferCUDATestHelper xfer;
    xfer.copy(output, device_output, cudaMemcpyDeviceToHost);
  }
#endif

  auto weighted = dynamic_cast<WeightedTimeSeries*>(output.get());
  ASSERT_NE(weighted,nullptr);
  ASSERT_EQ(weighted->get_ndat_per_weight(),nsamp_per_packet);

  static constexpr float limit_fp32 = 0.00001;

  const uint64_t ndat = output->get_ndat();
  const unsigned nchan = output->get_nchan();
  const unsigned npol = output->get_npol();
  const unsigned ndim = output->get_ndim();
  const unsigned nbit = input->get_nbit();
  const uint64_t nheaps = ndat / nsamp_per_packet;
  const size_t nval_per_packet = nsamp_per_packet * ndim;

  unsigned errors = 0;
  uint32_t denormalised_scale = 1;
  float expected = 0;

  for (uint32_t ichan=0; ichan < nchan; ichan++)
  {
    for (uint32_t ipol=0; ipol < npol; ipol++)
    {
      float* ptr = output->get_datptr(ichan, ipol);
      uint32_t odx = 0;
      for (uint64_t idat=0; idat < ndat; idat++)
      {
        for (uint32_t idim=0; idim < ndim; idim++)
        {
          uint32_t ival = odx % nval_per_packet;
          if (nbit == 8)
            expected = static_cast<float>(get_expected_data_8b(ival, ipol, denormalised_scale));
          else
            expected = static_cast<float>(get_expected_data_16b(ival, ipol, denormalised_scale));
          ASSERT_NEAR(ptr[odx], expected, limit_fp32);
          odx++;
        }
      }
    }

    uint16_t* weights = weighted->get_weights(ichan);
    uint16_t expected_weight = get_expected_weight(ichan);

    if (ichan > 0)
      ASSERT_NE(weights, weighted->get_weights(ichan-1));

    for (uint64_t iheap=0; iheap < nheaps; iheap++)
    {
      if (weights[iheap] != expected_weight)
      {
        std::cerr << "SKAParallelUnpackerTest::assert_output unexpected weight for ichan=" << ichan
             << " iheap=" << iheap << " have=" << weights[iheap] << " expected=" << expected_weight << std::endl;
        errors ++;
      }
    }
  }

  ASSERT_EQ(errors,0);
}

void SKAParallelUnpackerTest::set_input_output(std::shared_ptr<dsp::SKAParallelUnpacker> spu)
{
  spu->set_output_order(order);

  if (!on_gpu)
  {
    spu->set_input(input);
    spu->set_output(output);
  }
#ifdef HAVE_CUDA
  else
  {
    if (dsp::Operation::verbose)
      std::cerr << "Transferring ParallelBitSeries from Host to Device" << std::endl;
    TransferCUDATestHelper xfer;
    xfer.copy_bitseries(device_input->at(0), input->at(0), cudaMemcpyHostToDevice);
    xfer.copy_bitseries(device_input->at(1), input->at(1), cudaMemcpyHostToDevice);
    spu->set_input(device_input);
    spu->set_output(device_output);
  }
#endif
}

void SKAParallelUnpackerTest::call_configure(std::shared_ptr<dsp::SKAParallelUnpacker> spu)
{
  if (dsp::Operation::verbose)
    std::cerr << "spu->configure(obs_data)" << std::endl;
  spu->configure(obs_data);
}

void SKAParallelUnpackerTest::call_prepare(std::shared_ptr<dsp::SKAParallelUnpacker> spu)
{
  if (dsp::Operation::verbose)
    std::cerr << "spu->call_prepare()" << std::endl;
  spu->prepare();
}

void SKAParallelUnpackerTest::call_reserve(std::shared_ptr<dsp::SKAParallelUnpacker> spu)
{
  if (dsp::Operation::verbose)
    std::cerr << "spu->call_reserve()" << std::endl;
  spu->reserve();
}

void SKAParallelUnpackerTest::call_operate(std::shared_ptr<dsp::SKAParallelUnpacker> spu)
{
  if (dsp::Operation::verbose)
    std::cerr << "spu->call_operate()" << std::endl;
  spu->operate();
}

void SKAParallelUnpackerTest::call_reset(std::shared_ptr<dsp::SKAParallelUnpacker> spu)
{
  if (dsp::Operation::verbose)
    std::cerr << "spu->call_reset()" << std::endl;
  spu->reset();
}

dsp::SKAParallelUnpacker* SKAParallelUnpackerTest::new_device_under_test()
{
  Reference::To<dsp::SKAParallelUnpacker> device = new dsp::SKAParallelUnpacker;
#ifdef HAVE_CUDA
  if (on_gpu)
  {
    auto engine = new CUDA::SKAParallelUnpackerEngine(stream);
    engine->setup(device);
    device->set_engine(engine);
  }
#endif

  return device.release();
}

TEST_P(SKAParallelUnpackerTest, test_order_validity)
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());
  ASSERT_NO_THROW(set_input_output(spu));
  spu = nullptr;
  order = TimeSeries::OrderTFP;
  spu = std::make_shared<dsp::SKAParallelUnpacker>();
  ASSERT_ANY_THROW(set_input_output(spu));
}

TEST_P(SKAParallelUnpackerTest, test_construct_delete) // NOLINT
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());
  ASSERT_NE(spu, nullptr);
  spu = nullptr;
  ASSERT_EQ(spu, nullptr);
}

TEST_P(SKAParallelUnpackerTest, test_set_input_output) // NOLINT
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());
  ASSERT_NO_THROW(set_input_output(spu));
  spu = nullptr;
}

TEST_P(SKAParallelUnpackerTest, test_configure) // NOLINT
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());
  set_input_output(spu);
  ASSERT_NO_THROW(call_configure(spu));
  spu = nullptr;
}

TEST_P(SKAParallelUnpackerTest, test_prepare) // NOLINT
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());
  set_input_output(spu);
  call_configure(spu);
  ASSERT_NO_THROW(call_prepare(spu));
  spu = nullptr;
}

TEST_P(SKAParallelUnpackerTest, test_reserve) // NOLINT
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());
  set_input_output(spu);
  call_configure(spu);
  call_prepare(spu);
  ASSERT_NO_THROW(call_reserve(spu));
  spu = nullptr;
}

TEST_P(SKAParallelUnpackerTest, test_operate) // NOLINT
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());
  set_input_output(spu);
  call_configure(spu);
  call_prepare(spu);
  call_reserve(spu);
  ASSERT_NO_THROW(call_operate(spu));
  spu = nullptr;
}

TEST_P(SKAParallelUnpackerTest, test_transform) // NOLINT
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());

  set_input_output(spu);
  call_configure(spu);
  call_prepare(spu);
  call_reserve(spu);
  call_operate(spu);
  ASSERT_NO_THROW(assert_output());

  if (dsp::Operation::verbose)
    std::cerr << "spu->reset()" << std::endl;
  spu->reset();
  spu = nullptr;
}

TEST_P(SKAParallelUnpackerTest, test_reset) try // NOLINT
{
  std::shared_ptr<dsp::SKAParallelUnpacker> spu(new_device_under_test());

  set_input_output(spu);
  call_configure(spu);
  call_prepare(spu);
  call_reserve(spu);
  call_operate(spu);
  ASSERT_NO_THROW(call_reset(spu));
  spu = nullptr;
}
catch(Error& error)
{
  std::cerr << "SKAParallelUnpackerTest::test_reset exception" << error << std::endl;
  throw error;
}

std::vector<dsp::test::TestParam> get_test_params()
{
  std::vector<dsp::test::TestParam> params{};

  for (auto on_gpu : get_gpu_flags())
  {
    params.push_back({ on_gpu, std::string("low") });
    params.push_back({ on_gpu, std::string("mid_8b") });
    params.push_back({ on_gpu, std::string("mid_16b") });
  }
  return params;
}

INSTANTIATE_TEST_SUITE_P(SKAParallelUnpackerTestSuite, SKAParallelUnpackerTest,
  testing::ValuesIn(get_test_params()),
  [](const testing::TestParamInfo<SKAParallelUnpackerTest::ParamType>& info)
  {
    auto param = info.param;
    std::string name;
    if (param.on_gpu)
      name = "on_gpu_" + param.cbf_name;
    else
      name = "on_cpu_" + param.cbf_name;

    return name;
  }
); // NOLINT

} // namespace dsp::test
