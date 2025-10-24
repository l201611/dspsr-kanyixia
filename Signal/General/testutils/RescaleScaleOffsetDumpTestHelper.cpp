/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/RescaleScaleOffsetDumpTestHelper.h"
#include "ascii_header.h"

#include <iostream>
#include <cstdio>
#include <gtest/gtest.h>

void dsp::test::RescaleScaleOffsetDumpTestHelper::assert_file(std::string filename)
{
  static constexpr uint32_t header_size = 4096;

  FILE* file = std::fopen(filename.c_str(), "r");
  ASSERT_NE(file, nullptr);

  std::vector<char> header;
  header.resize(header_size, 0);
  char *buffer = header.data();

  ssize_t bytes_read = std::fread(buffer, sizeof(char), header_size, file);
  ASSERT_EQ(bytes_read, header_size);

  // extract critical meta-data from the DADAFile
  float file_hdr_version{0}, file_freq{0}, file_bw{0}, file_tsamp{0};
  uint32_t file_hdr_size, file_nchan{0}, file_npol{0}, file_ndim{0}, file_resolution{0},
    file_block_header_bytes{0}, file_block_data_bytes{0};
  int32_t file_nbit{0};
  char instrument[64];

  ASSERT_EQ(ascii_header_get(buffer, "HDR_SIZE", "%u", &file_hdr_size), 1);
  ASSERT_EQ(ascii_header_get(buffer, "HDR_VERSION", "%f", &file_hdr_version), 1);
  ASSERT_EQ(ascii_header_get(buffer, "NCHAN", "%u", &file_nchan), 1);
  ASSERT_EQ(ascii_header_get(buffer, "NPOL", "%u", &file_npol), 1);
  ASSERT_EQ(ascii_header_get(buffer, "NDIM", "%u", &file_ndim), 1);
  ASSERT_EQ(ascii_header_get(buffer, "NBIT", "%d", &file_nbit), 1);
  ASSERT_EQ(ascii_header_get(buffer, "FREQ", "%f", &file_freq), 1);
  ASSERT_EQ(ascii_header_get(buffer, "BW", "%f", &file_bw), 1);
  ASSERT_EQ(ascii_header_get(buffer, "TSAMP", "%f", &file_tsamp), 1);
  ASSERT_EQ(ascii_header_get(buffer, "RESOLUTION", "%u", &file_resolution), 1);
  ASSERT_EQ(ascii_header_get(buffer, "BLOCK_HEADER_BYTES", "%u", &file_block_header_bytes), 1);
  ASSERT_EQ(ascii_header_get(buffer, "BLOCK_DATA_BYTES", "%u", &file_block_data_bytes), 1);
  ASSERT_EQ(ascii_header_get(buffer, "INSTRUMENT", "%s", &instrument), 1);

  // assert that the values extracted from the DADAFile match the input
  auto npol = records[0].scales.size();
  auto nchan = records[0].scales[0].size();
  auto tsamp = 1e6/records[0].observation.get_rate();

  // the factor of 2 here is because there is the sample offset as well as the number of samples
  // used to calculate scales and offsets, both of which are uint64 values.
  auto expected_block_header_bytes = 2 * sizeof(uint64_t);

  // the factor of 2 here is to account for the scale and offset value both being in the record
  // and both are float32 values.
  auto expected_block_data_bytes = npol * nchan * 2 * sizeof(float);
  auto expected_resolution = expected_block_header_bytes + expected_block_data_bytes;

  ASSERT_EQ(header_size, file_hdr_size) << " mismatch in HDR_SIZE: expected=" << header_size << " output=" << file_hdr_size;
  ASSERT_EQ(1.0, file_hdr_version) << " mismatch in HDR_VERSION: expected=1.0 output=" << file_nbit;
  ASSERT_EQ(nchan, file_nchan) << " mismatch in NCHAN: expected=" << nchan << " output=" << file_nchan;
  ASSERT_EQ(npol, file_npol) << " mismatch in NPOL: expected=" << npol << " output=" << file_npol;
  ASSERT_EQ(-32, file_nbit) << " mismatch in NBIT: expected=-32" << " output=" << file_nbit;
  ASSERT_EQ(2, file_ndim) << " mismatch in NDIM: expected=2" << " output=" << file_ndim;
  ASSERT_EQ(expected_block_header_bytes, file_block_header_bytes) << " mismatch in BLOCK_HEADER_BYTES: expected=" << expected_block_header_bytes << " output=" << file_block_header_bytes;
  ASSERT_EQ(expected_block_data_bytes, file_block_data_bytes) << " mismatch in BLOCK_DATA_BYTES: expected=" << expected_block_data_bytes << " output=" << file_block_data_bytes;
  ASSERT_EQ(expected_resolution, file_resolution) << " mismatch in RESOLUTION: expected=" << expected_resolution << " output=" << file_resolution;
  ASSERT_EQ(tsamp, file_tsamp) << " expected TSAMP: expected=" << tsamp << " output=" << file_tsamp;
  ASSERT_EQ("dspsr", std::string(instrument)) << " expected INSTRUMENT: expected=dspsr" << " output=" << std::string(instrument);

  size_t bytes_to_read = expected_resolution * num_updates;

  std::vector<unsigned char> file_data;
  file_data.resize(bytes_to_read);

  bytes_read = std::fread(&file_data[0], sizeof(char), bytes_to_read, file);
  ASSERT_EQ(bytes_read, bytes_to_read);
  std::fclose(file);
  file = nullptr;

  ASSERT_EQ(bytes_read, static_cast<ssize_t>(bytes_to_read)) << " read fewer bytes of data than expected";

  unsigned char* data = file_data.data();
  for (auto &record : records)
  {
    auto file_sample_offset = reinterpret_cast<uint64_t *>(data)[0];
    data += sizeof(uint64_t);
    auto file_num_samples = reinterpret_cast<uint64_t *>(data)[0];
    data += sizeof(uint64_t);

    ASSERT_EQ(record.sample_offset, file_sample_offset) << " expected sample_offset=" << record.sample_offset << " output=" << file_sample_offset;
    ASSERT_EQ(record.num_samples, file_num_samples) << " expected num_samples=" << record.num_samples << " output=" << file_num_samples;

    for (auto ichan = 0; ichan < nchan; ichan++)
    {
      for (auto ipol = 0; ipol < npol; ipol++)
      {
        auto expected_offset = record.offsets[ipol][ichan];
        auto expected_scale = record.scales[ipol][ichan];

        auto file_scale = reinterpret_cast<float *>(data)[0];
        data += sizeof(float);
        auto file_offset = reinterpret_cast<float *>(data)[0];
        data += sizeof(float);

        ASSERT_EQ(expected_offset, file_offset) << " expected offset=" << expected_offset << " output=" << file_offset << ", ichan=" << ichan << ", ipol=" << ipol;
        ASSERT_EQ(expected_scale, file_scale) << " expected scale=" << expected_scale << " output=" << file_scale << ", ichan=" << ichan << ", ipol=" << ipol;
      }
    }
  }
}

void dsp::test::RescaleScaleOffsetDumpTestHelper::rescale_update(dsp::Rescale::update_record record)
{
  records.push_back(record);
  num_updates++;
}
