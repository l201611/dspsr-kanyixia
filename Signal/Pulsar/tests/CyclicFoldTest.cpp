/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/CyclicFoldTest.h"
#include "dsp/ImpulseTrain.h"
#include "dsp/Superposition.h"

#include "dsp/GtestMain.h"

#ifdef HAVE_CUDA
#include "dsp/CyclicFoldEngineCUDA.h"
#include "dsp/TransferCUDATestHelper.h"
#endif

#include <algorithm>
#include <fstream>
#include <cassert>

using namespace std;

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

//! Initialize the current test parameters
void CyclicFoldTest::SetUp()
{
  dsp::Shape::verbose = dsp::Observation::verbose;

  // The Cyclic Fold engine manages its own memory on the device
  output_on_device = false;

  if (::testing::UnitTest::GetInstance()->current_test_info()->value_param() != nullptr)
  {
    if (dsp::Operation::verbose)
      cerr << "CyclicFoldTest::Setup with test parameters" << endl;
    param = GetParam();
  }

  set_on_gpu (param.on_gpu);
  set_up();
}

//! Reset the current test parameters to their default
void CyclicFoldTest::TearDown()
{
  if (Operation::verbose)
    cerr << "CyclicFoldTest::TearDown" << endl;

  tear_down();
  param = CyclicFoldTestParam();

  if (Operation::verbose)
    cerr << "CyclicFoldTest::TearDown done" << endl;
}

dsp::CyclicFold* CyclicFoldTest::new_device_under_test()
{
  if (Operation::verbose)
    cerr << "CyclicFoldTest::new_device_under_test" << endl;
    
  Reference::To<dsp::CyclicFold> fold = new dsp::CyclicFold;

  input->set_state(param.state);

  if (param.on_gpu)
  {
#ifdef HAVE_CUDA
    if (Operation::verbose)
      cerr << "CyclicFoldTest::new_device_under_test call CyclicFold::set_engine CUDA::CyclicFoldEngineCUDA" << endl;

    auto engine = new CUDA::CyclicFoldEngineCUDA(get_cuda_stream());
    fold->set_engine(engine);
#endif
  }
  else
  {
    if (Operation::verbose)
      cerr << "CyclicFoldTest::new_device_under_test call CyclicFold::set_engine CyclicFoldEngine" << endl;

    fold->set_engine(new CyclicFoldEngine);
  }

  if (Operation::verbose)
    cerr << "CyclicFoldTest::new_device_under_test call initialize_operation" << endl;

  initialize_operation(fold);

  if (Operation::verbose)
    cerr << "CyclicFoldTest::new_device_under_test return ptr=" << (void*) fold.ptr() << endl;

  return fold.release();
}

TEST_P(CyclicFoldTest, test_construct_delete) try // NOLINT
{
  Reference::To<CyclicFold> fold = new_device_under_test();
  ASSERT_NE(fold, nullptr);
  delete fold;
  ASSERT_EQ(fold, nullptr);

  if (Operation::verbose)
    cerr << "CyclicFoldTest test_construct_delete done" << endl;
}
catch(Error& error)
{
  FAIL() << error << endl;
}

/* Verify that
   1) CyclicFold::prepare throws an exception if there is insufficient information about the source
   2) CyclicFold::prepare does not throw an exception after setting the folding period
*/
TEST_P(CyclicFoldTest, test_prepare) try // NOLINT
{
  Reference::To<dsp::CyclicFold> fold = new_device_under_test();

  // nothing known about the source
  EXPECT_THROW(fold->prepare(), Error);

  fold->set_folding_period(1.0);
  fold->set_nlag(64);
  fold->prepare();
}
catch(Error& error)
{
  FAIL() << error << endl;
}

template<typename T, typename U>
void write(const string& filename, T& expect, U& result, unsigned offset)
{
  string header = "Re[result], Im[result], Re[expect], Im[expect], Re[diff], Im[diff]";
  cerr << "writing " << header << " to " << filename << endl;
  ofstream out (filename.c_str());
  out << "# " << header << endl;
  for (unsigned idat=0; idat < expect.size(); idat++)
  {
    auto diff = result[idat+offset] - expect[idat];
    out << idat << " " << result[idat+offset].real() << " " << result[idat+offset].imag()
        << " " << expect[idat].real() << " " << expect[idat].imag()
        << " " << diff.real() << " " << diff.imag() << endl;
  }
}

/*!
  This tests compares the computed cyclic spectrum of an impulse train
  to the theoretical expectation.
*/
TEST_P(CyclicFoldTest, test_operate_impulse_train) try // NOLINT
{
  // disable warning about sensible nbin
  dsp::Fold::bin_width_warning = false;

  Reference::To<CyclicFold> fold = new_device_under_test();

  double sampling_interval = 1e-3;  // fake millisecond

  // the phase bin width is equal to the sampling interval
  unsigned nbin = 32;
  fold->set_folding_period(nbin * sampling_interval);

  Reference::To<ImpulseTrain> train = new ImpulseTrain;

  // the output of the signal generator is the input to the CyclicFold
  train->set_output(input);

  unsigned nlag = nbin;
  fold->set_nlag(nlag);
  fold->set_nbin(nbin);

  // five pulse periods
  unsigned periods = 5;
  uint64_t ndat = periods * nbin + nlag;

  auto info = train->get_info();
  info->set_rate(1.0/sampling_interval);
  info->set_ndat(ndat);
  info->set_npol(1);
  info->set_nchan(1);

  train->set_block_size(ndat);
  train->set_period_samples(nbin);

  train->operate();

  ASSERT_EQ(input->get_npol(), 1);
  ASSERT_EQ(input->get_nchan(), 1);
  ASSERT_EQ(input->get_rate(), 1.0/sampling_interval);

  perform_operation(fold);

  unsigned expected_nchan = (nlag-1)*2;

  dsp::PhaseSeries* result = fold->get_result ();
  ASSERT_EQ(result->get_npol(), 1);
  ASSERT_EQ(result->get_nchan(), expected_nchan);
  ASSERT_EQ(result->get_nbin(), nbin);

  auto hits = result->get_hits();

  for (unsigned ichan=0; ichan < expected_nchan; ichan++)
  {
    auto data = result->get_datptr(ichan);

    for (unsigned ibin=0; ibin < nbin; ibin++)
    {
      // each phase bin should have at least `periods` hits
      ASSERT_GE(hits[ibin], periods);
      auto value = data[ibin] / hits[ibin];

      /*
        expect a delta function in each channel, with height equal to number
        of periods integrated
      */
      float expected = 0.0;
      if (ibin == 0)
        expected = 1.0;
      ASSERT_EQ(value, expected);
    }
  }
}
catch(Error& error)
{
  FAIL() << error << endl;
}

/*!
  This tests compares the computed cyclic spectrum of a superposition of
  impulse trains to the theoretical expectation.

  Both impulse trains have a period of 32 samples, and train2 is delayed
  with respect to train1 by 6 samples.

  The sampling interval and the profile bin width (folding period / nbin)
  are set equal to one fake millisecond, so that there is one time sample
  per phase bin in the output periodic spectrum.

  The theoretical expectation begins by considering the periodic correlation:

  C (phi, tau) = < { x(t+tau/2) x*(t-tau/2); phi(t)=phi } >

  For tau = 0, C(phi,tau) is just the average pulse profile, which in this case
  consists of two delta functions:

  1) a delta function at phi=0 created by train 1, and
  2) a delta function at phi=6/32 created by train2

  For tau = 6, C(phi,tau) has another delta function at phi=3/32, where the
  train1 impulse at t-tau/2 correlates with the train2 impulse at t+tau/2.

  Apart from these three points { C(0,0), C(6,0), C(3,6) }, all other values
  of C(phi,tau) are zero.

  Now take the Fourier transform along the tau dimension to yield the periodic
  spectrum, which is returned by CyclicFold::get_result. The delta functions
  at C(0,0) and C(6,0) become constant values at all frequencies, and the delta
  function at C(3,6) becomes a cosine with 6 cycles across the band.
*/
TEST_P(CyclicFoldTest, test_operate_impulse_train_sum) try // NOLINT
{
  // disable warning about sensible nbin
  dsp::Fold::bin_width_warning = false;

  Reference::To<CyclicFold> fold = new_device_under_test();

  double sampling_interval = 1e-3;  // fake millisecond

  // the phase bin width is equal to the sampling interval
  unsigned nbin = 32;
  fold->set_folding_period(nbin * sampling_interval);

  Reference::To<ImpulseTrain> train1 = new ImpulseTrain;
  train1->set_period_samples(nbin);
  float amp1 = 1.0;
  train1->set_amplitude(amp1);

  Reference::To<ImpulseTrain> train2 = new ImpulseTrain;
  train2->set_period_samples(nbin);
  float amp2 = 0.5;
  train2->set_amplitude(amp2);
  unsigned ndelay = 6;         // delay train2 wrt train1
  train2->set_next_sample(ndelay);

  Reference::To<Superposition> sum = new Superposition;
  sum->add(train1);
  sum->add(train2);

  // the output of the signal generator is the input to the CyclicFold
  sum->set_output(input);

  // avoid interference between pulse k and pulse k+1
  unsigned nlag = nbin - ndelay;
  fold->set_nlag(nlag);
  fold->set_nbin(nbin);

  // five pulse periods
  unsigned periods = 5;
  uint64_t ndat = periods * nbin + nlag;
  sum->set_block_size(ndat);

  auto info = sum->get_info();
  info->set_rate(1.0/sampling_interval);
  info->set_ndat(ndat);
  info->set_npol(1);
  info->set_nchan(1);

  sum->operate();

#if _DEBUG
  auto data = input->get_datptr();
  for (unsigned idat=0; idat < ndat; idat++)
  {
    cout << "input " << idat << " " << data[idat*2] << " " << data[idat*2+1] << endl;
  }
#endif

  ASSERT_EQ(input->get_npol(), 1);
  ASSERT_EQ(input->get_nchan(), 1);
  ASSERT_EQ(input->get_ndat(), ndat);
  ASSERT_EQ(input->get_rate(), 1.0/sampling_interval);

  perform_operation(fold);

  unsigned expected_nchan = (nlag-1)*2;

  // get the resulting periodic spectrum
  dsp::PhaseSeries* result = fold->get_result ();
  ASSERT_EQ(result->get_npol(), 1);
  ASSERT_EQ(result->get_nchan(), expected_nchan);
  ASSERT_EQ(result->get_nbin(), nbin);

  unsigned errors = 0;

  auto hits = result->get_hits();

  for (unsigned ichan=0; ichan < expected_nchan; ichan++)
  {
    auto data = result->get_datptr(ichan);

    for (unsigned ibin=0; ibin < nbin; ibin++)
    {
      // each phase bin should have at least `periods` hits
      ASSERT_GE(hits[ibin], periods);
      auto value = data[ibin] / hits[ibin];

      float expected = 0;

      if (ibin == 0)             // C(0,0) due to train1 * train1
        expected = amp1*amp1;
      else if (ibin == ndelay)   // C(delay,0) due to train2 * train2
        expected = amp2*amp2;
      else if (ibin*2 == ndelay) // C(delay/2,delay) due to train1 * train2
      {
        // The factor of 2 is due to implicit conjugate symmetry, and z+z*=2Re[z]
        expected = 2.0*amp1*amp2*cos((2.0*M_PI*ndelay*ichan)/expected_nchan);
      }

      constexpr float tolerance = 1e-6;
      if ( fabs(value - expected) > tolerance )
      {
        cerr << "ERROR: ichan=" << ichan << " ibin=" << ibin << " val=" << data[ibin] << " expected=" << expected << endl;
        errors ++;
      }
    }
  }

  ASSERT_EQ(errors, 0);

}
catch(Error& error)
{
  FAIL() << error << endl;
}

vector<dsp::test::CyclicFoldTestParam> get_test_params()
{
  vector<dsp::test::CyclicFoldTestParam> params;

  for (auto on_gpu : get_gpu_flags())
  {
#if CyclicFold_support_real_valued_input
    for (auto state : {Signal::Analytic, Signal::Nyquist})
    {
      params.push_back({on_gpu, state});
    }
#else
    params.push_back({on_gpu, Signal::Analytic});
#endif
  }

  return params;
}

string CyclicFoldTestParam::get_name()
{
  string name;
  if (on_gpu)
    name = "gpu";
  else
    name = "cpu";

  if (state == Signal::Analytic)
    name += "_complex";
  else
    name += "_real";

  return name;
}

INSTANTIATE_TEST_SUITE_P(
    CyclicFoldTestSuite, CyclicFoldTest,
    testing::ValuesIn(get_test_params()),
    [](const testing::TestParamInfo<CyclicFoldTest::ParamType> &info)
    {
      auto param = info.param;
      return param.get_name();
    }); // NOLINT

} // namespace dsp::test
