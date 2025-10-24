/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Andrew Jameson, Jesmigel Cantos and Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToQuantizeTest.h"
#include "dsp/KnownDataSource.h"
#include "dsp/DADAOutputFile.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/NormalSampleStats.h"
#include "dsp/GtestMain.h"
#include "ascii_header.h"

#include <BoxMuller.h>      // from PSRCHIVE / EPSIC
#include <debug.h>          // also from PSRCHIVE / EPSIC

#include <algorithm>
#include <sstream>
#include <random>
#include <fcntl.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#include <cuda.h>
#endif

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

std::vector<int> nbits{1, 2, 4, 8, 16};

LoadToQuantizeTest::LoadToQuantizeTest()
{
}

void LoadToQuantizeTest::SetUp()
{
  config.application_name = "LoadToQuantizeTest";
  output_filename = "/tmp/LoadToQuantizeTest_output_file.dada";
  weights_output_filename = "/tmp/LoadToQuantizeTest_weights_file.dada";
  scale_offset_output_filename = "/tmp/LoadToQuantizeTest_scales_offset_file.dada";

  if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
  {
    auto param = GetParam();
    output_nbit = param.output_nbit;
    order = param.order;
    on_gpu = param.on_gpu;
    use_wts = param.use_wts;
    use_median_mad = param.use_median_mad;
  }
  else
  {
    use_wts = false;
  }

  if (use_wts)
  {
    auto weighted = new dsp::WeightedTimeSeries;
    weighted->set_nchan_weight(nchan_weight);
    weighted->set_npol_weight(npol_weight);
    weighted->set_ndat_per_weight(ndat_per_weight);
    input = weighted;
  }
  else
  {
    input = new dsp::TimeSeries;
  }
}

void LoadToQuantizeTest::TearDown()
{
  remove_output_files();
}

void LoadToQuantizeTest::remove_output_files()
{
  std::remove(output_filename.c_str());
  std::remove(scale_offset_output_filename.c_str());

  if (use_wts)
  {
    std::remove(weights_output_filename.c_str());
  }
}

double LoadToQuantizeTest::unpack_value(const unsigned char *value, uint64_t outidx)
{
  if (output_nbit == 16)
    return static_cast<double>(*reinterpret_cast<const int16_t *>(value));

  if (output_nbit == 8)
    return static_cast<double>(*reinterpret_cast<const int8_t *>(value));

  const unsigned char mask = ((unsigned char)pow(2, output_nbit)) - 1;

  auto byte_sample_idx = outidx % (8 / output_nbit);
  auto bit_shift = byte_sample_idx * output_nbit;

  unsigned char bitshifted = (unsigned char)(*value >> bit_shift);
  int result = (int)(bitshifted & mask);

  if (output_nbit == 1)
  {
    // map 0 to -1
    result = 2 * result - 1;
  }
  else
  {
    uint8_t msb = 1 << (output_nbit - 1);
    if (result & msb)
      result |= ~mask;
  }

  return static_cast<double>(result);
}

void LoadToQuantizeTest::generate_data()
{
  switch (order)
  {
  case dsp::TimeSeries::OrderFPT:
    generate_fpt();
    break;

  case dsp::TimeSeries::OrderTFP:
  default:
    generate_tfp();
    break;
  }

  if (use_wts)
    generate_weights();
}

void LoadToQuantizeTest::generate_fpt()
{
  auto nval = ndat * ndim;
  auto total_samples = ndat * nchan * npol * ndim * num_iterations;
  data.resize(total_samples, 0.0);

  std::vector<float> rnd_data(nval, 0.0);
  time_t now = time(nullptr);
  BoxMuller bm(now);
  uint64_t idx = 0;

  for (auto iteration = 0; iteration < num_iterations; iteration++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      const float offset = static_cast<float>(ipol + 1);
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        const float scale = static_cast<float>(ichan + 1);

        // generate normally distributed noise with zero mean and unit variance
        std::generate(rnd_data.begin(), rnd_data.end(), bm);

        for (uint64_t ival=0; ival<nval; ival++, idx++)
        {
          data[idx] = (rnd_data[ival] * scale) + offset;
        }
      }
    }
  }
}

void LoadToQuantizeTest::generate_tfp()
{
  auto total_samples = ndat * nchan * npol * ndim * num_iterations;
  data.resize(total_samples, 0.0);

  time_t now = time(nullptr);
  BoxMuller bm(now);

  std::vector<float> rnd_data(total_samples, 0.0);
  std::generate(rnd_data.begin(), rnd_data.end(), bm);
  uint64_t ival = 0;

  for (auto iteration = 0; iteration < num_iterations; iteration++)
  {
    for (uint64_t idat=0; idat < ndat; idat++)
    {
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        const float scale = static_cast<float>(ichan + 1);
        for (unsigned ipol=0; ipol < npol; ipol++)
        {
          const float offset = static_cast<float>(ipol + 1);
          for (unsigned idim=0; idim < ndim; idim++, ival++)
          {
            data[ival] = (rnd_data[ival] * scale) + offset;
          }
        }
      }
    }
  }
}

/*
  This helper function sets the weights in each channel and polarization
  to periodically cycle over iweight, ipol+409, ichan, iweight, ...

  e.g. for ichan=32, ipol=0, the sequence (indexed by iweight) would start with

  0, 409, 32, 3, 409, 32, ...

  409 is simply a large prime number that distinguishes pol=0 from ichan=0.

  By including ichan and ipol in the sequence, the unit tests verify that the
  the weights for each channel and polarization are in the correct place after
  the FPT to TFP transpose that takes place when copying weights from a
  WeightedTimeSeries to a BitSeries.
*/
uint16_t expected_weight(unsigned ichan, unsigned ipol, uint64_t iweight)
{
  unsigned state = iweight % 3;
  switch (state)
  {
    case 0: return iweight;
    case 1: return ipol + 409;
    case 2: return ichan;
  }
  return 0;
}

void LoadToQuantizeTest::generate_weights()
{
  auto weighted = dynamic_cast<WeightedTimeSeries*>(input.get());
  if (!weighted)
    throw Error (InvalidState, "LoadToQuantizeTest::generate_weights",
                "input TimeSeries is not a WeightedTimeSeries");

  auto nweights = weighted->get_nweights();
  auto total_weights = num_iterations * nweights * nchan_weight * npol_weight;
  weights.resize(total_weights, 0);

  if (dsp::Operation::verbose)
  {
    std::cerr << "generate_weights ndat: " << ndat << std::endl;
    std::cerr << "generate_weights ndat_per_weight: " << ndat_per_weight << std::endl;
    std::cerr << "generate_weights npol_weight: " << npol_weight << std::endl;
    std::cerr << "generate_weights nchan_weight: " << nchan_weight << std::endl;
    std::cerr << "generate_weights nweights: " << nweights << std::endl;
  }

  /*
    WeightedTimeSeries stores weights in FPT order
    (LoadToQuantizeTest will write these weights to DADA files in TFP order)
  */
  uint64_t idx = 0;
  for (auto iteration = 0; iteration < num_iterations; iteration++)
  {
    for (unsigned ichan=0; ichan<nchan_weight; ichan++)
    {
      for (unsigned ipol=0; ipol<npol_weight; ipol++)
      {
        for (uint64_t iweight=0; iweight<nweights; iweight++, idx++)
        {
          weights[idx] = expected_weight(ichan, ipol, nweights * iteration + iweight);
          DEBUG("generate_weights ichan=" << ichan << " ipol=" << ipol << " iwt=" << iweight << " val=" << weights[idx]);
        }
      }
    }
  }
}

void LoadToQuantizeTest::assert_data_file_header(const std::string& filename)
{
  // open the DADAFile for reading
  fd = ::open(filename.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);

  // allocate local memory for the DADAFile's ASCII header
  std::vector<char> header(header_size, 0);
  char* buffer = &header[0];

  // read the ASCII Header from the DADAFile
  ssize_t bytes_read = ::read(fd, buffer, header_size);
  ASSERT_EQ(bytes_read, header_size);

  // extract critical meta-data from the DADAFile
  float file_hdr_version{0}, file_freq{0}, file_bw{0}, file_tsamp{0};
  uint32_t file_hdr_size, file_nchan{0}, file_npol{0}, file_ndim{0}, file_nbit{0} ;
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
  ASSERT_EQ(header_size, file_hdr_size) << " mismatch in HDR_SIZE: expected=" << header_size << " output=" << file_hdr_size;
  ASSERT_EQ(1.0, file_hdr_version) << " mismatch in HDR_VERSION: expected=1.0 output=" << file_nbit;
  ASSERT_EQ(number_of_channels_to_keep, file_nchan) << " mismatch in NCHAN: expected=" << number_of_channels_to_keep << " output=" << file_nchan;
  ASSERT_EQ(number_of_polarizations_to_keep, file_npol) << " mismatch in NPOL: expected=" << number_of_polarizations_to_keep << " output=" << file_npol;
  ASSERT_EQ(output_nbit, file_nbit) << " mismatch in NBIT: expected=" << output_nbit<< " output=" << file_nbit;
  ASSERT_EQ(input->get_ndim(), file_ndim) << " mismatch in NDIM: input=" << input->get_ndim() << " output=" << file_ndim;
  ASSERT_EQ(1e6/input->get_rate(), file_tsamp) << " expected TSAMP: input=" << 1000000.0/input->get_rate() << " output=" << file_tsamp;
}

void LoadToQuantizeTest::assert_data_file(const std::string& filename)
{
  // ensure the DADAFile header meta-data is correct
  assert_data_file_header(filename);

  auto nchan_out = number_of_channels_to_keep ? number_of_channels_to_keep : nchan;
  auto npol_out = number_of_polarizations_to_keep ? number_of_polarizations_to_keep : npol;

  // determine how many bytes should be read from the DADAFile
  uint64_t nval = ndat * nchan_out * npol_out * ndim * num_iterations;
  size_t bytes_to_read = nval * sizeof(int8_t) * output_nbit / 8;

  // read the data from the DADAFile into a local buffer, only supports 8-bit integers
  std::vector<unsigned char> file_data;
  file_data.resize(bytes_to_read);
  ssize_t bytes_read = ::read(fd, &file_data[0], bytes_to_read);
  ::close(fd);
  fd = -1;
  ASSERT_EQ(bytes_read, static_cast<ssize_t>(bytes_to_read)) << " read fewer bytes of data than expected";

  // prepare empty vectors for the sums and sum_sqs of nchan and npol
  // use the Wilford online algorithm to calc mean and variance
  std::vector<std::vector<double>> means(nchan_out);
  std::vector<std::vector<double>> variances(nchan_out);
  for (unsigned ichan=0; ichan < nchan_out; ichan++)
  {
    means[ichan].resize(npol_out, 0);
    variances[ichan].resize(npol_out, 0);
  }

  // integrate the sum and sum of the square of each value
  uint64_t idx = 0;
  int idx_stride = output_nbit == 16 ? 2 : 1;
  int sample_per_stride = output_nbit == 16 ? 1 : 8 / output_nbit;

  auto stat_iterations = rescale_constant ? 1 : num_iterations;
  double count = static_cast<double>(ndat * ndim * stat_iterations);
  // the -1 below is due having a sample, not population, variance and hence so we use a degree of freedom of 1
  double var_factor = 1.0 / static_cast<double>(count - 1);

  for (uint64_t idat=0; idat < ndat * stat_iterations; idat++)
  {
    for (unsigned ichan=0; ichan < nchan_out; ichan++)
    {
      for (unsigned ipol=0; ipol < npol_out; ipol++)
      {
        double *curr_mean = &means[ichan][ipol];
        double *curr_var = &variances[ichan][ipol];
        for (unsigned idim=0; idim < ndim; idim++)
        {
          // + 1 is needed as we need be 1-offset not 0-offset for the calculation.
          auto curr_sample_count = static_cast<double>(idat * npol + idim + 1);
          uint64_t file_idx = idx / sample_per_stride;
          const double val = unpack_value(&file_data[file_idx], idx);

          auto val_mean_diff = val - *curr_mean;
          *curr_mean += val_mean_diff / curr_sample_count;
          *curr_var += (val - *curr_mean) * val_mean_diff * var_factor;

          idx += idx_stride;
        }
      }
    }
  }

  // expected values for the mean, stddev and variance after the Rescale/Digitizer operation
  double expected_rescaled_mean = static_cast<double>(dsp::GenericVoltageDigitizer::get_default_digi_mean(output_nbit));
  double digi_scale = static_cast<double>(dsp::GenericVoltageDigitizer::get_default_digi_scale(output_nbit));
  double expected_rescaled_variance = digi_scale * digi_scale;
  // NBIT_OUT of 2 has noticeable digitisation effects.  This have was calculated empirically.
  if (output_nbit == 2)
  {
    expected_rescaled_variance = 0.8608;
  }

  NormalSampleStats stats;
  stats.set_ndat (count);
  stats.set_variance (expected_rescaled_variance);

  double error_of_mean = stats.get_sample_mean_stddev();
  double error_of_variance = stats.get_sample_variance_stddev();

  static constexpr double threshold = 9.0;
  double assert_near_tolerance_mean = threshold * error_of_mean;
  double assert_near_tolerance_variance = threshold * error_of_variance;

  for (unsigned ichan=0; ichan < nchan_out; ichan++)
  {
    for (unsigned ipol=0; ipol < npol_out; ipol++)
    {
      double mean = means[ichan][ipol];
      double variance = variances[ichan][ipol];

      auto mean_err = fabs(mean - expected_rescaled_mean);
      auto mean_err_sigma = mean_err / error_of_mean;

      auto var_err = fabs(variance - expected_rescaled_variance);
      auto var_err_sigma = var_err / error_of_variance;

      EXPECT_NEAR(mean, expected_rescaled_mean, assert_near_tolerance_mean)
        << "nchan=" << nchan << ", npol=" << npol << ", ndat=" << ndat << ", ndim=" << ndim
        << ", stat_iterations=" << stat_iterations << ", count=" << count
        << ", ichan=" << ichan << ", ipol=" << ipol
        << ", mean_err=" << mean_err
        << ", mean_err_sigma=" << mean_err_sigma
        << ", var_err=" << var_err
        << ", var_err_sigma=" << var_err_sigma;

      EXPECT_NEAR(variance, expected_rescaled_variance, assert_near_tolerance_variance)
        << "nchan=" << nchan << ", npol=" << npol << ", ndat=" << ndat << ", ndim=" << ndim
        << ", stat_iterations=" << stat_iterations << ", count=" << count
        << ", ichan=" << ichan << ", ipol=" << ipol
        << ", mean_err=" << mean_err
        << ", mean_err_sigma=" << mean_err_sigma
        << ", var_err=" << var_err
        << ", var_err_sigma=" << var_err_sigma;
    }
  }
}

void LoadToQuantizeTest::assert_weights_file_header(const std::string& filename)
{
  auto weighted = dynamic_cast<WeightedTimeSeries*>(input.get());
  if (!weighted)
    throw Error (InvalidState, "LoadToQuantizeTest::assert_weights_file_header",
                "input TimeSeries is not a WeightedTimeSeries");

  // open the DADAFile for reading
  fd = ::open(filename.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);

  // allocate local memory for the DADAFile's ASCII header
  std::vector<char> header(header_size, 0);
  char* buffer = &header[0];

  // read the ASCII Header from the DADAFile
  ssize_t bytes_read = ::read(fd, buffer, header_size);
  ASSERT_EQ(bytes_read, header_size);

  // extract critical meta-data from the DADAFile
  float file_hdr_version{0}, file_freq{0}, file_bw{0}, file_tsamp{0};
  uint32_t file_hdr_size, file_nchan{0}, file_npol{0}, file_ndim{0}, file_nbit{0};
  ASSERT_EQ(ascii_header_get(buffer, "HDR_SIZE", "%u", &file_hdr_size), 1);
  ASSERT_EQ(ascii_header_get(buffer, "HDR_VERSION", "%f", &file_hdr_version), 1);
  ASSERT_EQ(ascii_header_get(buffer, "NCHAN", "%u", &file_nchan), 1);
  ASSERT_EQ(ascii_header_get(buffer, "NPOL", "%u", &file_npol), 1);
  ASSERT_EQ(ascii_header_get(buffer, "NDIM", "%u", &file_ndim), 1);
  ASSERT_EQ(ascii_header_get(buffer, "NBIT", "%u", &file_nbit), 1);
  ASSERT_EQ(ascii_header_get(buffer, "FREQ", "%f", &file_freq), 1);
  ASSERT_EQ(ascii_header_get(buffer, "BW", "%f", &file_bw), 1);
  ASSERT_EQ(ascii_header_get(buffer, "TSAMP", "%f", &file_tsamp), 1);

  unsigned expected_nchan = (number_of_channels_to_keep) ? number_of_channels_to_keep : weighted->get_nchan_weight();
  unsigned expected_npol = weighted->get_npol_weight();
  unsigned expected_nbit = 16; // weights are 16-bit values
  unsigned expected_ndim = 1;  // weights are shared by Re/Im
  double expected_tsamp = weighted->get_ndat_per_weight() * 1e6/input->get_rate();

  // assert that the values extracted from the DADAFile match the input
  ASSERT_EQ(header_size, file_hdr_size) << " mismatch in HDR_SIZE: expected=" << header_size << " output=" << file_hdr_size;
  ASSERT_EQ(1.0, file_hdr_version) << " mismatch in HDR_VERSION: expected=1.0 output=" << file_nbit;
  ASSERT_EQ(expected_nchan, file_nchan) << " mismatch in NCHAN: expected=" << expected_nchan << " output=" << file_nchan;
  ASSERT_EQ(expected_npol, file_npol) << " mismatch in NPOL: expected=" << expected_npol << " output=" << file_npol;
  ASSERT_EQ(expected_nbit, file_nbit) << " mismatch in NBIT: expected=" << expected_nbit << " output=" << file_nbit;
  ASSERT_EQ(expected_ndim, file_ndim) << " mismatch in NDIM: expected=" << expected_ndim << " output=" << file_ndim;
  ASSERT_EQ(expected_tsamp, file_tsamp) << " expected TSAMP: expected=" << expected_tsamp << " output=" << file_tsamp;
}

void LoadToQuantizeTest::assert_weights_file(const std::string& filename)
{
  // ensure the DADAFile header meta-data is correct
  assert_weights_file_header(filename);

  auto weighted = dynamic_cast<WeightedTimeSeries*>(input.get());
  if (!weighted)
    throw Error (InvalidState, "LoadToQuantizeTest::assert_weights_file",
                "input TimeSeries is not a WeightedTimeSeries");

  uint64_t expected_nweight = weighted->get_nweights();
  unsigned expected_nchan = (number_of_channels_to_keep) ? number_of_channels_to_keep : weighted->get_npol_weight();
  unsigned expected_npol = weighted->get_npol_weight();
  unsigned expected_nbit = 16; // weights are 16-bit values
  unsigned expected_ndim = 1;  // weights are shared by Re/Im

  if (dsp::Operation::verbose)
  {
    std::cerr << "LoadToQuantizeTest::assert_weights_file" << std::endl;
    std::cerr << "nweight=" << expected_nweight << " nchan=" << expected_nchan << " npol=" << expected_npol << std::endl;
    std::cerr << "start_channel_index=" << start_channel_index << std::endl;
    std::cerr << "number_of_channels_to_keep=" << number_of_channels_to_keep << std::endl;
    std::cerr << "number_of_polarizations_to_keep=" << number_of_polarizations_to_keep << std::endl;
  }

  // determine how many bytes should be read from the DADAFile
  uint64_t bits_to_read = expected_nchan * expected_npol * expected_nbit * expected_ndim * expected_nweight * num_iterations;
  ssize_t bytes_to_read = bits_to_read / 8;

  // read the weights from the DADAFile into a local buffer
  // header has already been read so we don't need an offset
  std::vector<unsigned char> file_data (bytes_to_read);
  ssize_t bytes_read = ::read(fd, file_data.data(), bytes_to_read);
  ::close(fd);
  fd = -1;
  ASSERT_EQ(bytes_read, bytes_to_read) << " read fewer bytes of data than expected";

  auto data = reinterpret_cast<uint16_t*>(file_data.data());

  // Weights are written to file in TFP order
  uint64_t errors = 0;
  for (uint64_t iwt=0; iwt<expected_nweight * num_iterations; iwt++)
  {
    for (unsigned ichan=0; ichan<expected_nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<expected_npol; ipol++)
      {
        auto expected = expected_weight(ichan, ipol, iwt);
        if (*data != expected)
        {
          std::cerr << "iwt=" << iwt << " ichan=" << ichan << " ipol=" << ipol << " expect=" << expected << " got=" << *data << std::endl;
          errors ++;
        }
        data++;
      }
    }
  }

  ASSERT_EQ(errors,0);
}

void LoadToQuantizeTest::prepare_config()
{
  {
    std::ostringstream oss;
    oss << start_channel_index<< ":" << ((start_channel_index + number_of_channels_to_keep) - 1);
    config.channel_range = oss.str();
  }

  {
    std::ostringstream oss;
    oss << start_polarization_index << ":" << ((start_polarization_index + number_of_polarizations_to_keep) - 1);
    config.pol_range = oss.str();
  }
  config.output_nbit = output_nbit;

  config.scale_offset_filename = scale_offset_output_filename;
  config.use_median_mad = use_median_mad;
  config.rescale_constant = rescale_constant;
  if (rescale_interval > 0.0)
  {
    config.rescale_interval = rescale_interval;
  }

  remove_output_files();
}

void LoadToQuantizeTest::prepare_input()
{
  static constexpr double tsamp_us = 64;
  static constexpr double day = 12345;
  static constexpr double ss = 54;
  static constexpr double fs = 0.222;
  MJD epoch(day, ss, fs);

  input->set_centre_frequency(300.0);
  input->set_bandwidth(-50.0);
  input->set_rate(1e6 / tsamp_us);
  input->set_start_time(epoch);
  input->set_nbit(32);
  input->set_nchan(nchan);
  input->set_npol(npol);
  input->set_ndim(ndim);
  input->set_order(order);
  input->resize(ndat);
}

void LoadToQuantizeTest::execute_engine() try
{
  dsp::DADAOutputFile sink(output_filename);

  // instantiate the engine with the pipeline configuration
  dsp::LoadToQuantize engine(&sink, &config);

  if (use_wts)
  {
    engine.set_output_weights(new dsp::DADAOutputFile(weights_output_filename));
  }

  // prepare the test source to deliver the specified number of iterations to the pipeline
  dsp::test::KnownDataSource source(num_iterations);
  source.set_output_order(order);
  source.set_data(data);
  if (use_wts)
    source.set_weights(weights);

  // connect the input time series to the source
  source.set_output(input);

  // execute the Load to Quantize pipeline
  engine.set_source(&source);

  engine.construct();

  scale_offset_file_helper = new dsp::test::RescaleScaleOffsetDumpTestHelper;
  engine.add_rescale_callback_handler(scale_offset_file_helper.get(), &dsp::test::RescaleScaleOffsetDumpTestHelper::rescale_update);

  engine.prepare();
  engine.run();
  engine.finish();

  auto metrics = engine.get_performance_metrics();
  if (dsp::Operation::verbose)
  {
    std::cerr << "LoadToQuantizeTest::execute_engine: finished processing" << std::endl;
    std::cerr << "LoadToQuantize Performance Metrics:" << std::endl;
    std::cout << "Total processing time: " << metrics->total_processing_time << " seconds" << std::endl;
    std::cout << "Total time spanned by the data: " << metrics->total_data_time << " seconds"  << std::endl;
    std::cout << "Total bytes processed: " << metrics->total_bytes_processed << std::endl;
  }
  ASSERT_TRUE(metrics->total_processing_time > 0.0) << "Total processing time should be greater than zero";
  ASSERT_TRUE(metrics->total_data_time > 0.0) << "Total data time should be greater than zero";
  ASSERT_TRUE(metrics->total_bytes_processed > 0) << "Total bytes processed should be greater than zero";
}
catch (Error& error)
{
  std::cerr << "LoadToQuantizeTest::execute_engine exception " << error << std::endl;
  throw error;
}

TEST_P(LoadToQuantizeTest, test_construct_delete) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif
  dsp::DADAOutputFile sink(output_filename);

  // prepare the LoadToQuantize::Config with default parameters
  prepare_config();

  dsp::LoadToQuantize* engine = nullptr;
  ASSERT_NO_THROW(engine = new dsp::LoadToQuantize(&sink, &config));

  delete engine;
}

TEST_P(LoadToQuantizeTest, test_process) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif

  number_of_channels_to_keep = nchan;
  number_of_polarizations_to_keep = npol;

  // prepare the LoadToQuantize::Config with default parameters
  prepare_config();

  // configure the input container
  prepare_input();

  // generate the input container
  generate_data();

  // prepare and execute the pipeline for 1 iteration
  ASSERT_NO_THROW(execute_engine());

  // assert the data in the file has the expected mean and variance
  assert_data_file(output_filename);

  if (use_wts)
  {
    // assert that the weights DADA file is as expected
    assert_weights_file(weights_output_filename);
  }

  scale_offset_file_helper->assert_file(scale_offset_output_filename);
}

TEST_P(LoadToQuantizeTest, test_process_multiple_blocks_const_rescale) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif

  number_of_channels_to_keep = nchan;
  number_of_polarizations_to_keep = npol;
  // The pipeline will process a total number of samples equal to num_iterations * ndat
  num_iterations = 3;
  rescale_constant = true;

  // prepare the LoadToQuantize::Config with default parameters
  prepare_config();

  // configure the input container
  prepare_input();

  // generate the input container with TFP ordered data
  generate_data();

  // prepare and execute the pipeline for 3 iterations
  ASSERT_NO_THROW(execute_engine());

  // assert the data in the file has the expected mean and variance
  assert_data_file(output_filename);

  if (use_wts)
  {
    // assert that the weights DADA file is as expected
    assert_weights_file(weights_output_filename);
  }

  ASSERT_EQ(1, scale_offset_file_helper->get_num_updates());
  scale_offset_file_helper->assert_file(scale_offset_output_filename);
}

TEST_P(LoadToQuantizeTest, test_process_multiple_blocks_const_rescale_with_interval) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif

  number_of_channels_to_keep = nchan;
  number_of_polarizations_to_keep = npol;
  // The pipeline will process a total number of samples equal to num_iterations * ndat
  num_iterations = 5;
  rescale_constant = true;
  // this is 2 * ndat * tsamp
  rescale_interval = 0.131072; // NOLINT
  // 1 update for first iteration and another to get the full rescale interval
  uint64_t expected_num_scloffs_updates = 2;

  // prepare the LoadToQuantize::Config with default parameters
  prepare_config();

  // configure the input container
  prepare_input();

  // generate the input container with TFP ordered data
  generate_data();

  // prepare and execute the pipeline for 3 iterations
  ASSERT_NO_THROW(execute_engine());

  // assert the data in the file has the expected mean and variance
  assert_data_file(output_filename);

  if (use_wts)
  {
    // assert that the weights DADA file is as expected
    assert_weights_file(weights_output_filename);
  }

  ASSERT_EQ(expected_num_scloffs_updates, scale_offset_file_helper->get_num_updates());
  scale_offset_file_helper->assert_file(scale_offset_output_filename);
}

TEST_P(LoadToQuantizeTest, test_process_multiple_blocks_nonconst_rescale) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif

  number_of_channels_to_keep = nchan;
  number_of_polarizations_to_keep = npol;
  // The pipeline will process a total number of samples equal to num_iterations * ndat
  num_iterations = 3;
  rescale_constant = false;

  // prepare the LoadToQuantize::Config with default parameters
  prepare_config();

  // configure the input container
  prepare_input();

  // generate the input container with TFP ordered data
  generate_data();

  // prepare and execute the pipeline for 3 iterations
  ASSERT_NO_THROW(execute_engine());

  // assert the data in the file has the expected mean and variance
  assert_data_file(output_filename);

  if (use_wts)
  {
    // assert that the weights DADA file is as expected
    assert_weights_file(weights_output_filename);
  }

  ASSERT_EQ(num_iterations, scale_offset_file_helper->get_num_updates());
  scale_offset_file_helper->assert_file(scale_offset_output_filename);
}

TEST_P(LoadToQuantizeTest, test_process_multiple_blocks_nonconst_rescale_with_interval) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif


  number_of_channels_to_keep = nchan;
  number_of_polarizations_to_keep = npol;
  // The pipeline will process a total number of samples equal to num_iterations * ndat
  num_iterations = 5;

  rescale_constant = false;
  // this is 2 * ndat * tsamp
  rescale_interval = 0.131072; // NOLINT
  uint64_t expected_num_scloffs_updates = 3;

  // prepare the LoadToQuantize::Config with default parameters
  prepare_config();

  // configure the input container
  prepare_input();

  // generate the input container with TFP ordered data
  generate_data();

  // prepare and execute the pipeline for 3 iterations
  ASSERT_NO_THROW(execute_engine());

  // assert the data in the file has the expected mean and variance
  assert_data_file(output_filename);

  if (use_wts)
  {
    // assert that the weights DADA file is as expected
    assert_weights_file(weights_output_filename);
  }

  ASSERT_EQ(expected_num_scloffs_updates, scale_offset_file_helper->get_num_updates());
  scale_offset_file_helper->assert_file(scale_offset_output_filename);
}

TEST_P(LoadToQuantizeTest, test_process_channel_subset) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif

  start_channel_index = 0;
  number_of_channels_to_keep = nchan - 2;
  number_of_polarizations_to_keep = npol;
  // The pipeline will process a total number of samples equal to num_iterations * ndat
  num_iterations = 2;

  // prepare the LoadToQuantize::Config with default parameters
  prepare_config();

  // configure the input container
  prepare_input();

  // generate the input container with TFP ordered data
  generate_data();

  // prepare and execute the pipeline for 2 iterations
  ASSERT_NO_THROW(execute_engine());

  // assert the data in the file has the expected mean and variance
  assert_data_file(output_filename);

  if (use_wts)
  {
    // assert that the weights DADA file is as expected
    assert_weights_file(weights_output_filename);
  }
}

TEST_P(LoadToQuantizeTest, test_process_polarization_subset) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif

  number_of_channels_to_keep = nchan;
  start_polarization_index = 0;
  number_of_polarizations_to_keep = npol - 1;
  // The pipeline will process a total number of samples equal to num_iterations * ndat
  num_iterations = 2;

  // prepare the LoadToQuantize::Config with default parameters
  prepare_config();

  // configure the input container
  prepare_input();

  // generate the input container with TFP ordered data
  generate_data();

  // prepare and execute the pipeline for 2 iterations
  ASSERT_NO_THROW(execute_engine());

  // assert the data in the file has the expected mean and variance
  assert_data_file(output_filename);

  if (use_wts)
  {
    // assert that the weights DADA file is as expected
    assert_weights_file(weights_output_filename);
  }
}

TEST_P(LoadToQuantizeTest, test_process_nchanpol_less_than_32) // NOLINT
{
#ifdef HAVE_CUDA
  if (on_gpu)
  {
     config.set_cuda_device("0");
  }
#endif

  // The pipeline will process a total number of samples equal to num_iterations * ndat
  num_iterations = 1;
  // this is to help speed up the tests
  ndat = 32;
  for (number_of_channels_to_keep = 1; number_of_channels_to_keep < 32; number_of_channels_to_keep++)
  {
    for (number_of_polarizations_to_keep = 1; number_of_polarizations_to_keep <= npol; number_of_polarizations_to_keep++)
    {
      if (number_of_channels_to_keep * number_of_polarizations_to_keep >= 32) {
        continue;
      }
      start_channel_index = 0;
      start_polarization_index = 0;

      // prepare the LoadToQuantize::Config with default parameters
      prepare_config();

      // configure the input container
      prepare_input();

      // generate the input container with TFP ordered data
      generate_data();

      // prepare and execute the pipeline for 2 iterations
      ASSERT_NO_THROW(execute_engine());
    }
  }

}

TEST_F(LoadToQuantizeTest, test_improper_configuration) // NOLINT
{
  // test improper channel configuration
  {
    number_of_channels_to_keep = nchan + 1;
    prepare_config();
    prepare_input();
    generate_data();
    ASSERT_THROW(execute_engine(), Error);
  }

  {
    number_of_channels_to_keep = 0;
    prepare_config();
    prepare_input();
    generate_data();
    ASSERT_THROW(execute_engine(), Error);
  }

  {
    config.channel_range = "blue:cheese";
    prepare_input();
    generate_data();
    ASSERT_THROW(execute_engine(), Error);
  }

  {
    config.channel_range = "-200:200";
    prepare_input();
    generate_data();
    ASSERT_THROW(execute_engine(), Error);
  }

  start_channel_index = 0;
  number_of_channels_to_keep = nchan;

  // test improper polarisation configuration
  {
    number_of_polarizations_to_keep = npol + 1;
    prepare_config();
    prepare_input();
    generate_data();
    ASSERT_THROW(execute_engine(), Error);
  }

  {
    number_of_polarizations_to_keep = 0;
    prepare_config();
    prepare_input();
    generate_data();
    ASSERT_THROW(execute_engine(), Error);
  }

  start_polarization_index = 0;
  number_of_polarizations_to_keep = npol;

  {
    for (auto nbit = 0; nbit <= 20; nbit++)
    {
      if (std::find(nbits.begin(), nbits.end(), nbit) != nbits.end())
        continue;

      output_nbit = nbit;
      prepare_config();
      prepare_input();
      generate_data();
      ASSERT_THROW(execute_engine(), Error);
    }
  }
}

std::vector<dsp::test::LoadToQuantizeTestParam> get_test_parameters() {
  std::vector<dsp::test::LoadToQuantizeTestParam> params{};
  std::vector<bool> wts_options = {false, true};
  std::vector<bool> median_mad_options = {false, true};
  std::vector<dsp::TimeSeries::Order> order_options = {dsp::TimeSeries::OrderTFP, dsp::TimeSeries::OrderFPT};

  for (auto output_nbit : nbits)
  {
    for (auto on_gpu: get_gpu_flags())
    {
      for (auto use_wts : wts_options)
      {
        for (auto use_median_mad : median_mad_options )
        {
          for (auto order : order_options)
          {
            params.push_back({ output_nbit, order, on_gpu, use_wts, use_median_mad });
          }
        }
      }
    }
  }

  return params;
}

INSTANTIATE_TEST_SUITE_P(
    LoadToQuantizeTestSuite, LoadToQuantizeTest,
    testing::ValuesIn(get_test_parameters()),
    [](const testing::TestParamInfo<LoadToQuantizeTest::ParamType> &info)
    {
      auto output_nbit = info.param.output_nbit;
      auto order = info.param.order;
      bool on_gpu = info.param.on_gpu;
      bool use_wts = info.param.use_wts;
      bool use_median_mad = info.param.use_median_mad;

      std::string name;

      if (order == dsp::TimeSeries::OrderFPT)
        name += "fpt_";
      else
        name += "tfp_";

      name += + "nbit" + std::to_string(std::abs(output_nbit));

      if (on_gpu)
        name += "_on_gpu";
      else
        name += "_on_cpu";

      if (use_median_mad)
        name += "_median_mad";
      else
        name += "_mean_std";

      if (use_wts)
        name += "_use_wts";
      else
        name += "_no_wts";

      return name;
    }); // NOLINT

} // namespace dsp::test
