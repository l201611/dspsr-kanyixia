/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ResponseTest.h"
#include "dsp/ResponseTestHelper.h"
#include "dsp/Observation.h"
#include "dsp/GtestMain.h"
#include "Spinor.h" // from epsic

#include <algorithm>
#include <iostream>
#include <random>
#include <cassert>

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

ResponseTest::ResponseTest()
{
  dsp::Shape::verbose = dsp::Observation::verbose;
}

dsp::Response* ResponseTest::new_device_under_test()
{
  Reference::To<dsp::Response> device = new dsp::Response;

  return device.release();
}

TEST_F(ResponseTest, test_construct_delete) // NOLINT
{
  auto response = new_device_under_test();
  ASSERT_NE(response, nullptr);
  delete response;
}

/* Verify that
   1) Response::prepare does not call Response::build,
   2) the first call to Response::match does call Reponse::build, and
   3) subsequent calls to Response::match do not call Reponse::build */
TEST_F(ResponseTest, test_prepare_match) // NOLINT
{
  ResponseTestHelper response;
  ASSERT_EQ(response.get_build_counter(), 0);

  Observation* fake_obs = nullptr;
  response.prepare(fake_obs);
  ASSERT_EQ(response.get_build_counter(), 0);

  response.match(fake_obs);
  ASSERT_EQ(response.get_build_counter(), 1);

  for (unsigned i=0; i<3; i++)
  {
    response.match(fake_obs);
    ASSERT_EQ(response.get_build_counter(), 1);
  }
}

/*
  This test verifies that the dimensions and data are correct
  after setting a complex-valued response function.
*/
TEST_F(ResponseTest, test_set_complex) // NOLINT
{
  Reference::To<Response> response = new_device_under_test();

  unsigned ndat = 1024;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  response->set(phasors);
  verify_equality(phasors, response);
}

/*
  This test verifies that the dimensions and data are correct
  after setting a matrix-valued response function.
*/
TEST_F(ResponseTest, test_set_matrix) // NOLINT
{
  Reference::To<Response> response = new_device_under_test();

  unsigned ndat = 256;
  std::vector<Jones<float>> matrices (ndat);
  create_test_vector(matrices);
  response->set(matrices);
  verify_equality(matrices, response);
}

/*
  This test verifies that the dimensions and data are correct
  after copying a complex-valued response function.
*/
TEST_F(ResponseTest, test_copy_complex) // NOLINT
{
  Reference::To<Response> other = new_device_under_test();

  unsigned ndat = 1024;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  other->set(phasors);

  Reference::To<Response> response = new_device_under_test();
  response->copy(other);
  verify_equality(phasors, response);
}

/*
  This test verifies that the dimensions and data are correct
  after copying a matrix-valued response function.
*/
TEST_F(ResponseTest, test_copy_matrix) // NOLINT
{
  Reference::To<Response> other = new_device_under_test();

  unsigned ndat = 256;
  std::vector<Jones<float>> matrices (ndat);
  create_test_vector(matrices);
  other->set(matrices);

  Reference::To<Response> response = new_device_under_test();
  response->copy(other);
  verify_equality(matrices, response);
}

/*
  This test verifies that a complex-valued response function
  is correctly applied to a spectrum.
*/
TEST_F(ResponseTest, test_operate_complex) // NOLINT
{
  Reference::To<Response> response = new_device_under_test();

  unsigned ndat = 1024;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  response->set(phasors);

  std::vector<std::complex<float>> spectrum (phasors);
  std::vector<std::complex<float>> expected_result (ndat);
  for (unsigned idat=0; idat<ndat; idat++)
  {
    expected_result[idat] = phasors[idat] * spectrum[idat];
  }

  response->operate(reinterpret_cast<float*>(spectrum.data()));

  verify_equality(spectrum, expected_result);
}

/*
  This test verifies that a matrix-valued response function
  is correctly applied to a pair of spectra.
*/
TEST_F(ResponseTest, test_operate_matrix) // NOLINT
{
  Reference::To<Response> response = new_device_under_test();

  unsigned ndat = 256;
  std::vector<Jones<float>> matrices (ndat);
  create_test_vector(matrices);
  response->set(matrices);

  std::vector<std::complex<float>> polA (ndat);
  create_test_vector(polA);
  std::vector<std::complex<float>> polB (ndat);

  std::vector<std::complex<float>> expected_polA (ndat);
  std::vector<std::complex<float>> expected_polB (ndat);

  unsigned shift = ndat/4;
  for (unsigned idat=0; idat<ndat; idat++)
  {
    // just to make polB different
    polB[idat] = polA[(idat+shift)%ndat];

    Spinor<float> evec(polA[idat], polB[idat]);
    Spinor<float> result = matrices[idat] * evec;

    expected_polA[idat] = result.x;
    expected_polB[idat] = result.y;
  }

  auto Afloat = reinterpret_cast<float*>(polA.data());
  auto Bfloat = reinterpret_cast<float*>(polB.data());

  response->operate(Afloat, Bfloat);

  verify_equality(polA, expected_polA);
  verify_equality(polB, expected_polB);
}

/*
  This test verifies that it is not possible to multiply a scalar inplace by a matrix
*/
TEST_F(ResponseTest, test_multiply_scalar_by_matrix) // NOLINT
{
  Reference::To<Response> response = new_device_under_test();

  unsigned ndat = 1024;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  response->set(phasors);

  Reference::To<Response> other = new_device_under_test();
  std::vector<Jones<float>> matrices (ndat);
  create_test_vector(matrices);
  other->set(matrices);

  ASSERT_THROW(response->multiply(other), Error);
}

/*
  This test verifies that a matrix is correctly multiplied in place by a scalar
*/
TEST_F(ResponseTest, test_multiply_matrix_by_scalar) // NOLINT
{
  Reference::To<Response> response = new_device_under_test();

  unsigned ndat = 1024;
  std::vector<Jones<float>> matrices (ndat);
  create_test_vector(matrices);
  response->set(matrices);

  Reference::To<Response> other = new_device_under_test();
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  other->set(phasors);

  response->multiply(other);

  for (unsigned idat=0; idat < ndat; idat++)
  {
    matrices[idat] *= phasors[idat];
  }

  verify_equality(matrices, response);
}

TEST_F(ResponseTest, test_downsampling_combining) // NOLINT
{
  ResponseTestHelper response;
  Observation input;

  response.set_downsampling(true);
  unsigned output_nchan = 1;

  input.set_oversampling_factor(Rational(4, 3));
  input.set_nchan(256);
  response.set_impulse_neg(7616);
  response.set_impulse_pos(6176);
  response.prepare(&input, output_nchan);

  // smallest multiples of 3*256 greater than original values
  ASSERT_EQ(response.get_impulse_neg(), 7680);
  ASSERT_EQ(response.get_impulse_pos(), 6912);

  input.set_nchan(2);
  response.set_impulse_neg(2);
  response.set_impulse_pos(2);
  response.prepare(&input, output_nchan);

  // smallest multiples of 3*2 greater than original values
  ASSERT_EQ(response.get_impulse_neg(), 6);
  ASSERT_EQ(response.get_impulse_pos(), 6);
}

} // namespace dsp::test
