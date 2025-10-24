/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/RescaleScaleOffsetDump.h"
#include "dsp/ASCIIObservation.h"

#include <fstream>
#include <cstring>
#include <algorithm>

using namespace std;

dsp::RescaleScaleOffsetDump::RescaleScaleOffsetDump(const std::string& filename)
{
  output_filename = filename;
}

void dsp::RescaleScaleOffsetDump::write_header (const dsp::Rescale::update_record& record)
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleScaleOffsetDump::write_header" << endl;

  vector<char> fname (FILENAME_MAX);
  char* filename = fname.data();

  MJD epoch = record.observation.get_start_time();
  if (!epoch.datestr( filename, FILENAME_MAX, output_filename.c_str() ))
  {
    throw Error (FailedSys, "dsp::RescaleScaleOffsetDump::write_header",
      "error MJD::datestr(" + output_filename + ")");
  }

  if (Operation::verbose)
  {
    cerr << "dsp::RescaleScaleOffsetDump::write_header filename=" << filename << endl;
  }

  FILE* ptr = std::fopen(filename, "w");
  if (!ptr)
  {
    throw Error(FailedSys, "dsp::RescaleScaleOffsetDump::write_header", "fopen(%s)", filename);
  }

  output = ptr;

  ASCIIObservation ascii (record.observation);
  ascii.set_machine ("dspsr");
  // using float32 for the data.
  ascii.set_nbit(-32);
  // we have scale and offset as the dimensions
  ascii.set_ndim(2);

  // ensure we have the correct RESOLUTION, BLOCK_HEADER_BYTES and BLOCK_DATA_BYTES
  const auto npol = record.scales.size();
  const auto nchan = record.scales.at(0).size();
  // 2 dims to account for the scale and the offset
  static constexpr unsigned ndim = 2;
  static constexpr unsigned num_header_values = 2;

  auto block_header_bytes = num_header_values * sizeof(uint64_t);
  auto block_data_bytes = nchan * npol * ndim * sizeof(float);
  auto resolution = block_header_bytes + block_data_bytes;

  if (dada_header.size() == 0)
    dada_header.resize();

  char *buffer = dada_header.get_header();
  ascii.unload(buffer);

  if (ascii_header_set (buffer, "HDR_SIZE", "%d", dada_header.size()) < 0)
    throw Error (InvalidState, "dsp::RescaleScaleOffsetDump::write_header",
		 "failed to set HDR_SIZE in output file header");

  if (ascii_header_set (buffer, "BLOCK_HEADER_BYTES", "%d", block_header_bytes) < 0)
    throw Error (InvalidState, "dsp::RescaleScaleOffsetDump::write_header",
		 "failed to set BLOCK_HEADER_BYTES in output file header");

  if (ascii_header_set (buffer, "BLOCK_DATA_BYTES", "%d", block_data_bytes) < 0)
    throw Error (InvalidState, "dsp::RescaleScaleOffsetDump::write_header",
		 "failed to set BLOCK_DATA_BYTES in output file header");

  if (ascii_header_set (buffer, "RESOLUTION", "%d", resolution) < 0)
    throw Error (InvalidState, "dsp::RescaleScaleOffsetDump::write_header",
		 "failed to set RESOLUTION in output file header");

  std::fwrite (buffer, sizeof(char), dada_header.size(), output);
}

void dsp::RescaleScaleOffsetDump::handle_scale_offset_updated(dsp::Rescale::update_record record)
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleScaleOffsetDump::handle_scale_offset_updated" << endl;

  // if we haven't already, open the file and write the header
  if (!output)
    write_header(record);

  const auto npol = record.scales.size();
  const auto nchan = record.scales.at(0).size();
  const unsigned ndim = 2; // scales and offsets
  static constexpr unsigned num_header_values = 2;

  uint64_t buffer_size = num_header_values * sizeof(uint64_t) + ndim * nchan * npol * sizeof(float);
  if (buffer.size() < buffer_size)
  {
    buffer.resize(buffer_size);
    std::fill(buffer.begin(), buffer.end(), 0);
  }

  auto ptr = reinterpret_cast<uint64_t*>(buffer.data());
  // write the sample offset
  *ptr = record.sample_offset;
  ptr ++;

  // write the number of statistic samples
  *ptr = record.num_samples;
  ptr ++;

  auto data = reinterpret_cast<float*>(ptr);

  // write the order the data as [chan][pol]
  for (auto ichan = 0; ichan < nchan; ichan++)
  {
    for (auto ipol = 0; ipol < npol; ipol++)
    {
      *data = record.scales[ipol][ichan];
      data ++;
      *data = record.offsets[ipol][ichan];
      data ++;
    }
  }

  std::fwrite(buffer.data(), sizeof(char), buffer_size, output);
  std::fflush(output);
}
