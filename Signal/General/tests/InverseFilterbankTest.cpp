/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/InverseFilterbankTest.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankResponse.h"

#ifdef HAVE_CUDA
#include "dsp/InverseFilterbankEngineCUDA.h"
#endif

#include "dsp/SourceFactory.h"
#include "dsp/Scratch.h"
#include "dsp/GtestMain.h"

using namespace std;

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

//! Initialize the current test parameters
void InverseFilterbankTest::SetUp()
{
  dsp::Shape::verbose = dsp::Operation::verbose;

  if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
  {
    if (dsp::Operation::verbose)
      std::cerr << "InverseFilterbankTest::Setup with test parameters" << std::endl;
    param = GetParam();
  }

  set_on_gpu(param.on_gpu);
  set_up();
}

//! Reset the current test parameters to their default
void InverseFilterbankTest::TearDown()
{
  tear_down();
  param = InverseFilterbankTestParam();
}

// synthesis filterbank parameters in 'low' configuration of ska-pst-dsp-model/matlab/test_vector.m
constexpr unsigned output_nchan = 1;
constexpr unsigned input_fft_length = 1024;
constexpr unsigned input_fft_overlap = 128;

dsp::InverseFilterbank* InverseFilterbankTest::new_device_under_test()
{
  Reference::To<dsp::InverseFilterbank> synthesis = new dsp::InverseFilterbank;

  Rational input_oversampling { 4, 3 };

  synthesis->set_output_nchan(output_nchan);
  synthesis->set_input_fft_length(input_fft_length);

  if (param.on_gpu)
  {
#ifdef HAVE_CUDA

    if (Operation::verbose)
      cerr << "InverseFilterbankTest::new_device_under_test construct new InverseFilterbankEngineCUDA" << endl;
  
    auto engine = new CUDA::InverseFilterbankEngineCUDA(get_cuda_stream());
    synthesis->set_engine(engine);

    auto scratch = new Scratch;
    scratch->set_memory (device_memory);
    synthesis->set_scratch (scratch);

    if (Operation::verbose)
      cerr << "InverseFilterbankTest::new_device_under_test CUDA-specific configuration completed" << endl;

#endif
  }
  else
  {
    synthesis->set_engine(new InverseFilterbankEngineCPU);
    synthesis->set_scratch (new Scratch);
  }

  // InverseFilterbank will always have a response.
  auto response = new dsp::InverseFilterbankResponse;
  response->set_apply_deripple(false);
  response->set_input_overlap(input_fft_overlap);
  response->set_pfb_dc_chan(true);

  synthesis->set_response(response);

  initialize_operation(synthesis);
  return synthesis.release();
}

TEST_P(InverseFilterbankTest, test_construct_delete) try // NOLINT
{
  auto inverse_filterbank = new_device_under_test();
  ASSERT_NE(inverse_filterbank, nullptr);
  delete inverse_filterbank;
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

// from test_InverseFilterbank.cpp SECTION ("setting input also sets oversampling factor")
TEST_P(InverseFilterbankTest, test_set_input_sets_oversampling_factor) try // NOLINT
{
  Reference::To<InverseFilterbank> synthesis = new_device_under_test();

  Rational os_factor_old (8, 7);
  Rational os_factor_new (4, 3);

  synthesis->set_oversampling_factor(os_factor_old);
  ASSERT_EQ (synthesis->get_oversampling_factor(), os_factor_old);

  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  input->set_oversampling_factor(os_factor_new);
  synthesis->set_input (input);
  ASSERT_EQ (synthesis->get_oversampling_factor(), os_factor_new);
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

void InverseFilterbankTest::load_input_from_file (const std::string& filename)
{
  char* test_dir = getenv("DSPSR_TEST_DATA_DIR");
  if (!test_dir)
  {
    throw Error (InvalidState, "InverseFilterbankTest::load_input_from_file",
                 "DSPSR_TEST_DATA_DIR environment variable is not defined");
  }

  string test_filename = test_dir;
  test_filename += "/inverse_filterbank/" + filename;
  
  SourceFactory factory;
  source = factory.create(test_filename);

  auto info = source->get_info();
  auto ndat = info->get_ndat();
  source->set_block_size(ndat);

  // the output of the Source is the input of the InverseFilterbank
  source->set_output(input);
  source->operate();
}

/*
  This test loads a test vector consisting of a single delta function that has
  passed through a Matlab implementation of an analysis polyphase filterbank.
  It inverts the polyphase filterbank (without derippling) and verifies that
  spurious power / temporal leakage is below an acceptable threshold.

  Matlab commands that generate test vector in ska-pst-dsp-model:

    test_vector(cbf='low', domain='temporal', Nstate=1, nbit=8);
    sgcht (input='../products/test_vector_spectral.dada', cfg='low');

  If the DSPSR_TEST_DATA_DIR environment variable is not set, then this test is skipped.
*/
TEST_P(InverseFilterbankTest, test_temporal_fidelity) // NOLINT
{
  char* test_dir = getenv("DSPSR_TEST_DATA_DIR");
  if (!test_dir)
  {
    GTEST_SKIP() << "DSPSR_TEST_DATA_DIR environment variable is not defined" << endl;
  }

  constexpr unsigned input_nchan = 256;
  // reported for first test state output by test_vector.m
  constexpr unsigned expected_impulse_idat = 12864;

  test_temporal_fidelity ("lowpsi_temporal_test_vector.dada", input_nchan, expected_impulse_idat);
}

/*
  This test loads a test vector consisting of a single delta function that has
  passed through a Matlab implementation of two stages of analysis polyphase filterbank.
  It inverts the polyphase filterbank (without derippling) and verifies that
  spurious power / temporal leakage is below an acceptable threshold.

  Matlab commands that generate test vector in ska-pst-dsp-model:

    test_vector(cbf='low', domain='temporal', critical=true, Ncoarse=256, Nstate=1, nbit=8);
    sgcht (input='../products/test_vector_temporal.dada', cfg='sps', cfg2='low', critical=true);

  If the DSPSR_TEST_DATA_DIR environment variable is not set, then this test is skipped.
*/
TEST_P(InverseFilterbankTest, test_temporal_fidelity_two_stages) // NOLINT
{
  char* test_dir = getenv("DSPSR_TEST_DATA_DIR");
  if (!test_dir)
  {
    GTEST_SKIP() << "DSPSR_TEST_DATA_DIR environment variable is not defined" << endl;
  }

  constexpr unsigned input_nchan = 256 * 216;
  // reported for first test state output by test_vector.m
  constexpr unsigned expected_impulse_idat = 2695680;

  test_temporal_fidelity ("sps_low_temporal_test_vector.dada", input_nchan, expected_impulse_idat);
}

void InverseFilterbankTest::test_temporal_fidelity (const std::string& filename, unsigned input_nchan, unsigned expected_impulse_idat) try
{
  Reference::To<InverseFilterbank> inverse_filterbank = new_device_under_test();

  load_input_from_file (filename);

  constexpr unsigned ndim_complex = 2;

  auto info = source->get_info();
  inverse_filterbank->set_input_fft_length(input_fft_length, info);

  // expect multi-channel complex-valued input
  ASSERT_EQ(input->get_nchan(), input_nchan);
  ASSERT_EQ(input->get_state(), Signal::Analytic);
  ASSERT_EQ(input->get_ndim(), ndim_complex);

  perform_operation(inverse_filterbank);

  // expect single-channnel complex-valued output
  ASSERT_EQ(output->get_nchan(), output_nchan);
  ASSERT_EQ(output->get_state(), Signal::Analytic);
  ASSERT_EQ(output->get_ndim(), ndim_complex);

  auto data = reinterpret_cast<complex<float>*>(output->get_datptr(0,0));

  double tot_spurious_power = 0.0;
  float max_spurious_power = 0.0;
  unsigned max_idat = 0;
  float impulse_power = 0.0;

  unsigned ndat = output->get_ndat();
  for (unsigned idat=0; idat < ndat; idat++)
  {
    float power = std::norm(data[idat]);
    if (idat == expected_impulse_idat)
    {
      impulse_power = power;
    }
    else
    {
      tot_spurious_power += power;
      if (power > max_spurious_power)
      {
        max_idat = idat;
        max_spurious_power = power;
      }
    }
  }

  ASSERT_GT(impulse_power, 0.0);

  double max_spurious_dB = 10.0 * log10(max_spurious_power / impulse_power);
  double tot_spurious_dB = 10.0 * log10(tot_spurious_power / impulse_power);

  if (Operation::verbose)
  {
    cerr << "impulse power=" << impulse_power << endl;
    cerr << "total spurious power=" << tot_spurious_power << endl;
    cerr << "maximum spurious power=" << max_spurious_power << endl;
    cerr << "maximum spurious power index=" << max_idat << endl;

    cerr << "tot. spurious dB=" << tot_spurious_dB << endl;
    cerr << "max. spurious dB=" << max_spurious_dB << endl;
  }

  // L2-4185 SKAO-CSP_Low_PST_REQ-697 CSP_Low.PST maximum temporal leakage: "-60 dB (power ratio) for larger temporal offsets"
  // L2-4554 SKAO-CSP_Mid_PST_REQ-386 CSP_Mid.PST maximum temporal leakage: "-60 dB (power ratio) for larger temporal offsets"
  constexpr double max_spurious_dB_threshold = -70.0; // test performed on 26 Jul 2025 achieved -72.3
  ASSERT_LT(max_spurious_dB, max_spurious_dB_threshold);

  // L2-4184 SKAO-CSP_Low_PST_REQ-721 CSP_Low.PST total temporal leakage: "less than -50 dB (power ratio)"
  // L2-4553 SKAO-CSP_Mid_PST_REQ-413 CSP_Mid.PST total temporal leakage: "less than -50 dB (power ratio)"
  constexpr double tot_spurious_dB_threshold = -68.0; // test performed on 26 Jul 2025 achieved -69.9
  ASSERT_LT(tot_spurious_dB, tot_spurious_dB_threshold);
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

/*
  This test loads a test vector consisting of a pure tone that has
  passed through a Matlab implementation of an analysis polyphase filterbank.
  It inverts the polyphase filterbank (without derippling) and verifies that
  spurious power / spectral leakage is below an acceptable threshold.

  If the DSPSR_TEST_DATA_DIR environment variable is not set, then this test is skipped.
*/
TEST_P(InverseFilterbankTest, test_spectral_fidelity) try // NOLINT
{
  char* test_dir = getenv("DSPSR_TEST_DATA_DIR");
  if (!test_dir)
  {
    GTEST_SKIP() << "DSPSR_TEST_DATA_DIR environment variable is not defined" << endl;
  }

  Reference::To<InverseFilterbank> inverse_filterbank = new_device_under_test();

  load_input_from_file ("lowpsi_spectral_test_vector.dada");

  constexpr unsigned input_nchan = 256;
  constexpr unsigned expected_tone_ichan = 98304;
  constexpr unsigned ndim_complex = 2;

  auto info = source->get_info();
  inverse_filterbank->set_input_fft_length(input_fft_length, info);

  unsigned freq_resolution = inverse_filterbank->get_frequency_resolution();
  ASSERT_GT(freq_resolution, 0);

  // expect multi-channel complex-valued input
  ASSERT_EQ(input->get_nchan(), input_nchan);
  ASSERT_EQ(input->get_state(), Signal::Analytic);
  ASSERT_EQ(input->get_ndim(), ndim_complex);

  perform_operation(inverse_filterbank);

  // expect single-channnel complex-valued output
  ASSERT_EQ(output->get_nchan(), output_nchan);
  ASSERT_EQ(output->get_state(), Signal::Analytic);
  ASSERT_EQ(output->get_ndim(), ndim_complex);

  auto ndat = output->get_ndat();
  ASSERT_GE(ndat,freq_resolution);

  unsigned nfft = freq_resolution;
  vector<float> fft (nfft*ndim_complex, 0.0);
  FTransform::fcc1d (nfft, fft.data(), output->get_datptr(0,0));
  auto data = reinterpret_cast<complex<float>*>(fft.data());

  double tot_spurious_power = 0.0;
  float max_spurious_power = 0.0;
  unsigned max_idat = 0;
  float tone_power = 0.0;

  for (unsigned idat=0; idat < nfft; idat++)
  {
    float power = std::norm(data[idat]);
    if (idat == expected_tone_ichan)
    {
      tone_power = power;
    }
    else
    {
      tot_spurious_power += power;
      if (power > max_spurious_power)
      {
        max_idat = idat;
        max_spurious_power = power;
      }
    }
  }

  ASSERT_GT(tone_power, 0.0);

  double max_spurious_dB = 10.0 * log10(max_spurious_power / tone_power);
  double tot_spurious_dB = 10.0 * log10(tot_spurious_power / tone_power);

  if (Operation::verbose)
  {
    cerr << "Nfft=" << nfft << endl;
    cerr << "expected harmonic=" << expected_tone_ichan << endl;
    cerr << "tone power=" << tone_power << endl;
    cerr << "total spurious power=" << tot_spurious_power << endl;
    cerr << "maximum spurious power=" << max_spurious_power << endl;
    cerr << "maximum spurious power index=" << max_idat << endl;

    cerr << "tot. spurious dB=" << tot_spurious_dB << endl;
    cerr << "max. spurious dB=" << max_spurious_dB << endl;
  }
 
  // L2-4187 SKAO-CSP_Low_PST_REQ-627 CSP_Low.PST maximum spectral confusion: "no more than -60 dB (power ratio)"
  // L2-4556 SKAO-CSP_Mid_PST_REQ-385 CSP_Mid.PST maximum spectral confusion: "no more than -60 dB (power ratio)"
  constexpr double max_spurious_dB_threshold = -70.0; // test performed on 26 Jul 2025 achieved -72.3
  ASSERT_LT(max_spurious_dB, max_spurious_dB_threshold);

  // L2-4186 SKAO-CSP_Low_PST_REQ-722 CSP_Low.PST total spectral confusion: "less than -50 dB (power ratio)"
  // L2-4555 SKAO-CSP_Mid_PST_REQ-414 CSP_Mid.PST total spectral confusion: "no more than -50 dB (power ratio)"
  constexpr double tot_spurious_dB_threshold = -68.0; // test performed on 26 Jul 2025 achieved -69.9
  ASSERT_LT(tot_spurious_dB, tot_spurious_dB_threshold);
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

// from test_InverseFilterbank.cpp TEST_CASE ("InverseFilterbank runs on channelized data")
/*
  This is essentially a "smoke test" that passes if nothing breaks
  (e.g. no exceptions, no segfaults, ...)
*/
TEST_P(InverseFilterbankTest, test_can_apply_deripple) try // NOLINT
{
  char* test_dir = getenv("DSPSR_TEST_DATA_DIR");
  if (!test_dir)
  {
    GTEST_SKIP() << "DSPSR_TEST_DATA_DIR environment variable is not defined" << endl;
  }
  Reference::To<InverseFilterbank> inverse_filterbank = new_device_under_test();

  // copied from test_config.toml
  string filename = "channelized.simulated_pulsar.noise_0.0.nseries_3.ndim_2.dump";

  load_input_from_file (filename);

  auto info = source->get_info();
  inverse_filterbank->set_input_fft_length(input_fft_length, info);

  input->set_pfb_dc_chan(true);
  input->set_pfb_nchan(input->get_nchan());

  Reference::To<dsp::InverseFilterbankResponse> deripple = new dsp::InverseFilterbankResponse;

  ASSERT_GT(info->get_deripple().size(), 0);

  unsigned freq_res = 768;

  deripple->set_fir_filter(info->get_deripple()[0]);
  deripple->set_apply_deripple(false);
  deripple->set_ndat(freq_res);
  deripple->resize(1, 1, freq_res, 2);

  inverse_filterbank->set_response(deripple);
  inverse_filterbank->set_buffering_policy(NULL);
  inverse_filterbank->set_output_nchan(1);

  Reference::To<dsp::InverseFilterbankResponse> zero_DM_response = new dsp::InverseFilterbankResponse;
  zero_DM_response->copy(deripple);

  Reference::To<dsp::TimeSeries> zero_DM_output = new dsp::TimeSeries;

  inverse_filterbank->set_zero_DM(true);
  inverse_filterbank->set_zero_DM_output(zero_DM_output);
  inverse_filterbank->set_zero_DM_response(zero_DM_response);

  perform_operation(inverse_filterbank);

  ASSERT_EQ(zero_DM_response->get_ndat(), inverse_filterbank->get_response()->get_ndat());
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

std::vector<dsp::test::InverseFilterbankTestParam> get_test_params()
{
  std::vector<dsp::test::InverseFilterbankTestParam> params;

  for (auto on_gpu : get_gpu_flags())
  {
    params.push_back({on_gpu});
  }

  return params;
}

std::string InverseFilterbankTestParam::get_name()
{
  std::string name;
  if (on_gpu)
    name = "gpu";
  else
    name = "cpu";

  return name;
}

INSTANTIATE_TEST_SUITE_P(
    InverseFilterbankTestSuite, InverseFilterbankTest,
    testing::ValuesIn(get_test_params()),
    [](const testing::TestParamInfo<InverseFilterbankTest::ParamType> &info)
    {
      auto param = info.param;
      return param.get_name();
    }); // NOLINT

} // namespace dsp::test
