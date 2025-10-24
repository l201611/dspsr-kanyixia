/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/ConvolutionTest.h"
#include "dsp/ResponseTestHelper.h"
#include "dsp/TestResponse.h"
#include "dsp/Memory.h"
#include "dsp/GtestMain.h"

#ifdef HAVE_CUDA
#include "dsp/ConvolutionCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#endif

#include <algorithm>
#include <fstream>
#include <cassert>

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

ConvolutionTest::ConvolutionTest()
{
  response = new Response;
}

//! Initialize the current test parameters
void ConvolutionTest::SetUp()
{
  dsp::Shape::verbose = dsp::Observation::verbose;

  if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
  {
    if (dsp::Operation::verbose)
      std::cerr << "ConvolutionTest::Setup with test parameters" << std::endl;
    param = GetParam();
  }

  set_on_gpu (param.on_gpu);
  set_up();
}

//! Reset the current test parameters to their default
void ConvolutionTest::TearDown()
{
  tear_down();
  param = ConvolutionTestParam();
}

dsp::Convolution* ConvolutionTest::new_device_under_test()
{
  Reference::To<dsp::Convolution> convolution = new dsp::Convolution;
  convolution->set_response(response);

  input->set_state(param.state);

  if (param.on_gpu)
  {
#ifdef HAVE_CUDA
    auto engine = new CUDA::ConvolutionEngine(get_cuda_stream());
    convolution->set_engine(engine);
#endif
  }
  else
  {
    convolution->set_engine(nullptr);
  }

  initialize_operation(convolution);

  return convolution.release();
}

TEST_P(ConvolutionTest, test_construct_delete) try // NOLINT
{
  auto convolution = new_device_under_test();
  ASSERT_NE(convolution, nullptr);
  delete convolution;
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

/* Verify that
   1) Convolution::prepare does not call Response::build,
   2) the first call to Convolution::operate does call Response::build, and
   3) subsequent calls to Convolution::operate do not call Response::build */
TEST_P(ConvolutionTest, test_prepare) try // NOLINT
{
  Reference::To<dsp::Convolution> convolution = new_device_under_test();

  ResponseTestHelper response_tmp;
  ASSERT_EQ(response_tmp.get_build_counter(), 0);

  convolution->set_response(&response_tmp);

  convolution->prepare();
  ASSERT_EQ(response_tmp.get_build_counter(), 0);

  convolution->operate();
  ASSERT_EQ(response_tmp.get_build_counter(), 1);

  convolution->operate();
  ASSERT_EQ(response_tmp.get_build_counter(), 1);
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

template<typename T, typename U>
void write(const std::string& filename, T& expect, U& result, unsigned offset)
{
  std::string header = "Re[result], Im[result], Re[expect], Im[expect], Re[diff], Im[diff]";
  std::cerr << "writing " << header << " to " << filename << std::endl;
  std::ofstream out (filename.c_str());
  out << "# " << header << std::endl;
  for (unsigned idat=0; idat < expect.size(); idat++)
  {
    auto diff = result[idat+offset] - expect[idat];
    out << idat << " " << result[idat+offset].real() << " " << result[idat+offset].imag()
        << " " << expect[idat].real() << " " << expect[idat].imag()
        << " " << diff.real() << " " << diff.imag() << std::endl;
  }
}

// Verify that Convolution::operate yields the expected impulse response function
/*
  The impulses are placed in two different places in the two polarizations.
  The test confirms that the difference between the known impulse and the impulse
  generated through convolution (multiplication in the Fourier domain) are
  very close, and that all other samples are zero.
*/
TEST_P(ConvolutionTest, test_operate_tukey) try // NOLINT
{
  Reference::To<Convolution> convolution = new_device_under_test();

  auto resp = new TestResponse;

  unsigned n_neg = 128; // NOLINT
  resp->set_impulse_neg(n_neg);

  // odd-valued n_pos better tests the R->C correction
  unsigned n_pos = 256; // NOLINT
  resp->set_impulse_pos(n_pos);
  convolution->set_response(resp);

  convolution->prepare();
  unsigned nfilt = resp->get_ndat();

  if (dsp::Operation::verbose)
    std::cerr << "test_operate_tukey: n_neg=" << n_neg << " n_pos=" << n_pos << " -> optimal ndat=" << nfilt << std::endl;

  // number of FFTs to execute
  /* Currently, this large number will fail if batched FFTs are enabled in ConvolutionCUDA.h */
  unsigned n_loop = 256 + 13; // NOLINT

#ifdef HAVE_CUDA_BATCHED_FFTS_WORKING
  if (param.on_gpu)
  {
    n_loop = CUDA::ConvolutionEngine::maximum_batched_nfft / nfilt;
    std::cerr << "batching FFTs n_loop=" << n_loop << std::endl;
  }
#endif

  unsigned n_overlap = n_neg + n_pos;
  unsigned n_keep = nfilt - n_overlap;
  unsigned expected_ndat = n_keep * 2*n_loop;

  unsigned input_ndat = expected_ndat + n_overlap;
  unsigned nfloat = input_ndat * 2;

  if (input->get_state() == Signal::Nyquist)
    input_ndat *= 2;

  constexpr unsigned npol = 2;
  input->set_npol(npol);
  input->resize(input_ndat);
  input->zero();

  // pol0: delta function at first possible complete output
  // pol1: delta function in the overlap between FFTs
  unsigned base_offsets[2] = { n_overlap, nfilt-n_pos };
  std::vector<unsigned> offsets (n_loop * npol);

  for (unsigned iloop=0; iloop < n_loop; iloop++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      unsigned offset = offsets[iloop*npol+ipol] = base_offsets[ipol] + iloop*2*(nfilt-n_overlap) + iloop + ipol;

      assert(offset*2 < nfloat);

      // real-valued delta function
      float* data = input->get_datptr(0,ipol);
      data[offset*2] = 1.0;
    }
  }

  // perform the convolution operation
  perform_operation(convolution);
  ASSERT_EQ(output->get_ndat(), expected_ndat);

  // expected output at the expected offset
  auto& expect = resp->temporal;

  if (dsp::Operation::verbose)
  {
    // print out the first two pulses
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      auto result = reinterpret_cast<std::complex<float>*>(output->get_datptr(0,ipol));
      unsigned offset = offsets[ipol] - n_overlap;

      std::string filename = "test_operate_tukey_pol=" + tostring(ipol) + ".txt";
      write(filename,expect,result,offset);
    }
  }

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    double baseline_power = 0.0;
    uint64_t baseline_idat = 0;
    uint64_t baseline_count = 0;

    for (unsigned iloop=0; iloop < n_loop; iloop++)
    {
      unsigned offset = offsets[iloop*npol+ipol] - n_overlap;
      auto result = reinterpret_cast<std::complex<float>*>(output->get_datptr(0,ipol));

      /*
        Verify that the impulse output by convolution matches the known impulse response
      */
      double power = 0.0;
      double power_diff = 0.0;
      for (unsigned idat=0; idat < n_overlap; idat++)
      {
        auto diff = result[idat+offset] - expect[idat];
        power_diff += std::norm(diff);
        power += std::norm(expect[idat]);
      }

      double ratio = power_diff / power;
      double mean_error = sqrt(ratio / n_overlap);

      if (dsp::Operation::verbose)
        std::cerr << "test_operate_tukey mean_error=" << mean_error << std::endl;

      float threshold = 5e-8;
      if (mean_error > threshold)
      {
        std::string filename = "test_operate_tukey_fail.txt";
        write(filename,expect,result,offset);
      }

      ASSERT_LT(mean_error, threshold) << " ipol=" << ipol << " iloop=" << iloop << std::endl;

      /*
        Verify that the samples outside of the expected impulse are close to zero
      */
      unsigned end_idx = output->get_ndat();
      if (iloop + 1 < n_loop)
      {
        // stop at the start of the next pulse
        end_idx = offsets[(iloop+1)*npol+ipol] - n_overlap;
      }

      // add the power in off-pulse samples that precede the pulse
      while (baseline_idat < offset)
      {
        baseline_power += std::norm(result[baseline_idat]);
        baseline_idat++;
        baseline_count++;
      }

      // skip the pulse
      baseline_idat += n_overlap;

      // add the power in off-pulse samples that follow the pulse
      while (baseline_idat < end_idx)
      {
        baseline_power += std::norm(result[baseline_idat]);
        baseline_idat++;
        baseline_count++;
      }
    }

    double rms = sqrt(baseline_power / baseline_count);

    if (dsp::Operation::verbose)
      std::cerr << "test_operate_tukey rms=" << rms << std::endl;
    ASSERT_LT(rms, 5e-8) << " ipol=" << ipol << std::endl;
  }
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

std::vector<dsp::test::ConvolutionTestParam> get_test_params()
{
  std::vector<dsp::test::ConvolutionTestParam> params;

  for (auto on_gpu : get_gpu_flags())
  {
    for (auto state : {Signal::Analytic, Signal::Nyquist})
    {
      params.push_back({on_gpu, state});
    }
  }

  return params;
}

std::string ConvolutionTestParam::get_name()
{
  std::string name;
  if (on_gpu)
    name = "gpu";
  else
    name = "cpu";

  if (state == Signal::Analytic)
    name += "_complex";
  else
    name += "_real";

  if (matrix)
    name += "_matrix";
  else
    name += "_scalar";

  return name;
}

INSTANTIATE_TEST_SUITE_P(
    ConvolutionTestSuite, ConvolutionTest,
    testing::ValuesIn(get_test_params()),
    [](const testing::TestParamInfo<ConvolutionTest::ParamType> &info)
    {
      auto param = info.param;
      return param.get_name();
    }); // NOLINT

} // namespace dsp::test
