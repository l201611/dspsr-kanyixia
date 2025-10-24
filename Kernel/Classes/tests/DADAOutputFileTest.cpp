/***************************************************************************
 *
 *   Copyright (C) 2024 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADAOutputFile.h"
#include "dsp/GtestMain.h"
#include "dsp/BitSeries.h"
#include "dsp/DADAOutputFileTest.h"
#include "ascii_header.h"

#include <fcntl.h>

#include <algorithm>
#include <random>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>

//! main method passed to googletest
int main(int argc, char *argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test
{

  DADAOutputFileTest::DADAOutputFileTest()
  {
  }

  void DADAOutputFileTest::SetUp()
  {
    input = new dsp::BitSeries;
  }

  void DADAOutputFileTest::TearDown()
  {
    input = nullptr;
    std::remove(filename.c_str());
  }

  void DADAOutputFileTest::generate_random_data(int8_t* data_ptr, size_t data_length)
  {
    // Seed the random number generator
    static constexpr short min_int8 = -128;
    static constexpr short max_int8 = 127;
    std::uniform_int_distribution<short> distribution(min_int8, max_int8);
    for (uint64_t i = 0; i < data_length; i++)
    {
      data_ptr[i] = static_cast<int8_t>(distribution(generator));
    }
  }

  void DADAOutputFileTest::setup_random_data(unsigned num_iterations)
  {
    input->set_state(Signal::Analytic);
    input->set_ndim(ndim);
    input->set_npol(npol);
    input->set_nchan(nchan);
    input->set_nbit(nbit);
    input->resize(ndat);

    size_t required_size = input->get_size() * num_iterations;
    input_data.resize(required_size);

    generate_random_data(&input_data[0], required_size);
  }

  void DADAOutputFileTest::setup_known_data(unsigned num_iterations)
  {
    // adjust attributes to match known_input_data array
    ndim = 2;
    nbit = 8;
    nchan = 2;
    npol = 2;
    ndat = 4;

    input->set_state(Signal::Analytic);
    input->set_ndim(ndim);
    input->set_npol(npol);
    input->set_nchan(nchan);
    input->set_nbit(nbit);
    input->resize(ndat);

    // the known input vector is 32 values, which corresponds to the input dimensions hard-coded here.
    ASSERT_EQ(known_input_data.size(), input->get_size()) << " mismatch between test input parameters and known_input_data size";

    size_t required = input->get_size() * num_iterations;
    input_data.resize(required);

    size_t offset = 0;
    for (unsigned i=0; i<num_iterations; i++)
    {
      std::copy(known_input_data.begin(), known_input_data.end(), &input_data[offset]);
      offset += input->get_size();
    }
  }

  void DADAOutputFileTest::assert_header()
  {
    // open a DADA Output File
    fd = ::open(filename.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);

    std::vector<char> header(header_size, 0);
    char* buffer = &header[0];

    // read the ASCII Header from the DADAFile
    ssize_t bytes_read = ::read(fd, buffer, header_size);
    ASSERT_EQ(bytes_read, header_size);

    // extract critical meta-data from the DADAFile
    float file_hdr_version{0}, file_freq{0}, file_bw{0}, file_tsamp{0};
    uint32_t file_hdr_size{0}, file_nchan{0}, file_npol{0}, file_ndim{0}, file_nbit{0} ;
    ASSERT_EQ(ascii_header_get(buffer, "HDR_SIZE", "%u", &file_hdr_size), 1);
    ASSERT_EQ(ascii_header_get(buffer, "HDR_VERSION", "%f", &file_hdr_version), 1);
    ASSERT_EQ(ascii_header_get(buffer, "NCHAN", "%u", &file_nchan), 1);
    ASSERT_EQ(ascii_header_get(buffer, "NPOL", "%u", &file_npol), 1);
    ASSERT_EQ(ascii_header_get(buffer, "NDIM", "%u", &file_ndim), 1);
    ASSERT_EQ(ascii_header_get(buffer, "NBIT", "%u", &file_nbit), 1);
    ASSERT_EQ(ascii_header_get(buffer, "FREQ", "%f", &file_freq), 1);
    ASSERT_EQ(ascii_header_get(buffer, "BW", "%f", &file_bw), 1);
    ASSERT_EQ(ascii_header_get(buffer, "TSAMP", "%f", &file_tsamp), 1);

    // assert that the values extracted from the DADAFile match the input
    ASSERT_EQ(header_size, file_hdr_size) << " mismatch in HDR_SIZE: input=" << header_size << " output=" << file_hdr_size;
    ASSERT_EQ(1.0, file_hdr_version) << " mismatch in HDR_VERSION: input=" << 1.0 << " output=" << file_hdr_version;
    ASSERT_EQ(nchan, file_nchan) << " mismatch in NCHAN: input=" << nchan << " output=" << file_nchan;
    ASSERT_EQ(npol, file_npol) << " mismatch in NPOL: input=" << npol << " output=" << file_npol;
    ASSERT_EQ(ndim, file_ndim) << " mismatch in NDIM: input=" << ndim << " output=" << file_ndim;
    ASSERT_EQ(nbit, file_nbit) << " mismatch in NBIT: input=" << nbit << " output=" << file_nbit;
    ASSERT_EQ(freq, file_freq) << " mismatch in FREQ: input=" << freq << " output=" << file_freq;
    ASSERT_EQ(bw, file_bw) << " mismatch in BW: input=" << bw << " output=" << file_bw;
    ASSERT_EQ(tsamp, file_tsamp) << " mismatch TSAMP: input=" << tsamp << " output=" << file_tsamp;
  }

  void DADAOutputFileTest::assert_data()
  {
    // open the DADAFile and assert the header parameters match in input BitSeries.
    assert_header();

    std::vector<int8_t> file_data;
    size_t bytes_to_read = input_data.size();
    file_data.resize(bytes_to_read);
    ssize_t bytes_read = ::read(fd, &file_data[0], bytes_to_read);
    ::close(fd);
    fd = -1;
    ASSERT_EQ(bytes_read, static_cast<ssize_t>(bytes_to_read)) << " read fewer bytes of data than expected";

    for (unsigned idx=0; idx<file_data.size(); idx++)
    {
      const int8_t expected_value = input_data[idx];
      const int8_t actual_value = file_data[idx];
      ASSERT_EQ(expected_value, actual_value) << " idx=" << idx;
    }
  }

  bool DADAOutputFileTest::perform_transform(std::shared_ptr<dsp::DADAOutputFile> dof, unsigned num_iterations)
  {
    try
    {
      // set metadata related not related to data dimensions
      input->set_centre_frequency(freq);
      input->set_bandwidth(bw);
      input->set_rate(1e6 / tsamp);
      input->set_start_time(start_time);
      dof->set_input(input);
      dof->prepare();

      size_t input_size = input->get_size();
      size_t input_offset = 0;

      for (unsigned i=0; i<num_iterations; i++)
      {
        // set the input bit series to the prepared data values
        std::memcpy(
          reinterpret_cast<void *>(input->get_rawptr()),
          reinterpret_cast<void*>(&input_data[input_offset]),
          input_size
        );

        dof->operate();
        input_offset += input_size;
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
  }

  TEST_F(DADAOutputFileTest, test_construct_delete) // NOLINT
  {
    filename = "/tmp/DADAOutputFileTest_test_construct_delete.dada";
    std::remove(filename.c_str());
    std::shared_ptr<dsp::DADAOutputFile> dof = std::make_shared<dsp::DADAOutputFile>(filename);
    ASSERT_NE(dof, nullptr);
    dof = nullptr;
    ASSERT_EQ(dof, nullptr);
  }

  TEST_F(DADAOutputFileTest, test_random_data_large) // NOLINT
  {
    unsigned num_iterations = 2;
    nchan = 1024;
    ndat = 1024;
    filename = "/tmp/DADAOutputFileTest_test_random_data_large.dada";
    std::remove(filename.c_str());
    std::shared_ptr<dsp::DADAOutputFile> dof = std::make_shared<dsp::DADAOutputFile>(filename);
    ASSERT_NO_THROW(setup_random_data(num_iterations));
    ASSERT_TRUE(perform_transform(dof, num_iterations));
    assert_data();
  }

  TEST_F(DADAOutputFileTest, test_random_data) // NOLINT
  {
    unsigned num_iterations = 1;
    filename = "/tmp/DADAOutputFileTest_test_random_data.dada";
    std::remove(filename.c_str());
    std::shared_ptr<dsp::DADAOutputFile> dof = std::make_shared<dsp::DADAOutputFile>(filename);
    ASSERT_NO_THROW(setup_random_data(num_iterations));
    ASSERT_TRUE(perform_transform(dof, num_iterations));
    assert_data();
  }

  TEST_F(DADAOutputFileTest, test_known_data) // NOLINT
  {
    unsigned num_iterations = 1;
    filename = "/tmp/DADAOutputFile_test_known_data.dada";
    std::remove(filename.c_str());
    std::shared_ptr<dsp::DADAOutputFile> dof = std::make_shared<dsp::DADAOutputFile>(filename);
    ASSERT_NO_THROW(setup_known_data(num_iterations));
    ASSERT_TRUE(perform_transform(dof, num_iterations));
    assert_data();
  }

} // namespace dsp::test
