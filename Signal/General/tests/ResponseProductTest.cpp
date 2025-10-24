/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ResponseProductTest.h"
#include "dsp/ResponseTestHelper.h"
#include "dsp/Operation.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "dsp/GtestMain.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <cassert>

using namespace std;

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test {

ResponseProductTest::ResponseProductTest()
{
  dsp::Shape::verbose = dsp::Operation::verbose;
}

dsp::ResponseProduct* ResponseProductTest::new_device_under_test()
{
  return new dsp::ResponseProduct;
}

TEST_F(ResponseProductTest, test_construct_delete) // NOLINT
{
  auto response = new_device_under_test();
  ASSERT_NE(response, nullptr);
  delete response;
}

/*
  This test verifies that the dimensions and data are correct
  after setting a complex-valued response function.
*/
TEST_F(ResponseProductTest, test_set_complex) // NOLINT
{
  Reference::To<Response> response = new ResponseTestHelper;

  unsigned ndat = 1024;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  response->set(phasors);

  Reference::To<ResponseProduct> product = new_device_under_test();
  product->add_response(response);

  verify_equality(phasors,product);
}

/*
  This test verifies that the product of a scalar and a scalar is a scalar
*/
TEST_F(ResponseProductTest, test_multiply_scalar_by_scalar) // NOLINT
{
  Reference::To<Response> scalarA = new ResponseTestHelper;

  unsigned ndat = 1024;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  scalarA->set(phasors);

  Reference::To<Response> scalarB = new ResponseTestHelper;
  scalarB->set(phasors);

  Reference::To<ResponseProduct> product = new_device_under_test();
  product->add_response(scalarA);
  product->add_response(scalarB);
  
  for (unsigned idat=0; idat < ndat; idat++)
  {
    phasors[idat] *= phasors[idat];
  }

  verify_equality(phasors, product);
}

/*
  This test verifies that the product of a matrix and a scalar is a matrix
*/
TEST_F(ResponseProductTest, test_multiply_matrix_by_scalar) // NOLINT
{
  Reference::To<Response> matrix = new ResponseTestHelper;

  unsigned ndat = 1024;
  std::vector<Jones<float>> matrices (ndat);
  create_test_vector(matrices);
  matrix->set(matrices);

  Reference::To<Response> scalar = new ResponseTestHelper;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  scalar->set(phasors);

  Reference::To<ResponseProduct> product = new_device_under_test();
  product->add_response(scalar);
  product->add_response(matrix);

  for (unsigned idat=0; idat < ndat; idat++)
  {
    matrices[idat] *= phasors[idat];
  }

  verify_equality(matrices, product);
}

/*
  This test verifies that, after calling Response::prepare and after calling Response::match,
  the impulse_neg and impulse_pos attributes of the product are equal to the sums of the 
  the impulse_neg and impulse_pos attributes of the multiplicands.
*/
TEST_F(ResponseProductTest, test_impulse_pos_and_neg) // NOLINT
{
  Reference::To<Response> scalarA = new ResponseTestHelper;

  unsigned ndat = 1024;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  scalarA->set(phasors);

  constexpr unsigned negA = 5;
  constexpr unsigned posA = 7;
  scalarA->set_impulse_neg(negA);
  scalarA->set_impulse_pos(posA);

  Reference::To<Response> scalarB = new ResponseTestHelper;
  scalarB->set(phasors);

  constexpr unsigned negB = 11;
  constexpr unsigned posB = 13;
  scalarB->set_impulse_neg(negB);
  scalarB->set_impulse_pos(posB);

  Reference::To<ResponseProduct> product = new_device_under_test();
  product->add_response(scalarA);
  product->add_response(scalarB);

  Observation* fake_obs = nullptr;
  product->prepare(fake_obs);

  unsigned neg_expect = negA + negB;
  unsigned pos_expect = posA + posB;

  ASSERT_EQ(product->get_impulse_neg(), neg_expect);
  ASSERT_EQ(product->get_impulse_pos(), pos_expect);

  product->match(fake_obs);

  ASSERT_EQ(product->get_impulse_neg(), neg_expect);
  ASSERT_EQ(product->get_impulse_pos(), pos_expect);
}

/*
  This test verifies that, after calling Response::prepare and after calling Response::match,
  the specified frequency resolution is maintained.
*/
TEST_F(ResponseProductTest, test_set_frequency_resolution) // NOLINT
{
  Reference::To<Response> scalarA = new ResponseTestHelper;

  unsigned ndat = 1024;
  std::vector<std::complex<float>> phasors (ndat);
  create_test_vector(phasors);
  scalarA->set(phasors);

  constexpr unsigned negA = 5;
  constexpr unsigned posA = 7;
  scalarA->set_impulse_neg(negA);
  scalarA->set_impulse_pos(posA);

  Reference::To<Response> scalarB = new ResponseTestHelper;
  scalarB->set(phasors);

  constexpr unsigned negB = 11;
  constexpr unsigned posB = 13;
  scalarB->set_impulse_neg(negB);
  scalarB->set_impulse_pos(posB);

  Reference::To<ResponseProduct> product = new_device_under_test();
  product->add_response(scalarA);
  product->add_response(scalarB);

  /*
    verify that the optimal ndat that would be chosen by default is not coincidentally
    equal to the frequency resolution that will be set in the next step
  */

  if (Shape::verbose)
    cerr << "ResponseProductTest, test_set_frequency_resolution calling ResponseProduct::set_optimal_ndat" << endl;

  product->set_optimal_ndat();
  unsigned optimal_ndat = product->get_ndat();
  ASSERT_NE(optimal_ndat, ndat);

  /*
    verify that setting the frequency resolution of a multiplicand also sets the 
    frequency resolution of the product
  */

  if (Shape::verbose)
    cerr << "ResponseProductTest, test_set_frequency_resolution calling Response::set_frequency_resolution" << endl;

  scalarB->set_frequency_resolution(ndat);

  Observation* fake_obs = nullptr;

  product->prepare(fake_obs);
  ASSERT_EQ(product->get_ndat(), ndat);

  product->match(fake_obs);
  ASSERT_EQ(product->get_ndat(), ndat);
}

/*
  This test verifies the fix for bugs/128 by testing that the downsampling flag
  of the product is not reset to that of the first multiplicand copied.
  It also verifies that the downsampling flag is correctly interpreted, such that
  impulse_neg and impulse_pos are properly set to a multiple of the oversampling
  ratio denoninator times the number of channels inverted.
*/
TEST_F(ResponseProductTest, test_downsampling) try // NOLINT
{
  Reference::To<ResponseProduct> product = new_device_under_test();
  Reference::To<Response> multiplicand = new ResponseTestHelper;

  product->add_response(multiplicand);
  product->add_response(new ResponseTestHelper);

  Observation input;

  // the product is configured for downsampling
  product->set_downsampling(true);
  unsigned output_nchan = 1;

  input.set_oversampling_factor(Rational(4, 3));
  input.set_nchan(256);

  // the mulitiplicands determine impulse_pos and impulse_neg
  multiplicand->set_impulse_neg(7616);
  multiplicand->set_impulse_pos(6176);
  product->prepare(&input, output_nchan);

  // verify that the downsampling flag is still set to true
  ASSERT_EQ(product->get_downsampling(), true);

  /*
     verify that the downsampling flag was correctly interpreted
     and that impulse_pos and impulse_neg have been increased to 
     the smallest multiples of 3*256 greater than their original values
  */
  ASSERT_EQ(product->get_impulse_neg(), 7680);
  ASSERT_EQ(product->get_impulse_pos(), 6912);
}
catch(Error& error)
{
  FAIL() << error << std::endl;
}

/*
  This test verifies that, if a multiplicand with an optimal_fft strategy
  is added to the product, then the product adopts the strategy.
*/
TEST_F(ResponseProductTest, test_add_sets_optimal_fft) // NOLINT
{
  Reference::To<Response> scalarA = new ResponseTestHelper;
  Reference::To<Response> scalarB = new ResponseTestHelper;

  auto optimal = new OptimalFFT;

  scalarB->set_optimal_fft(optimal);

  Reference::To<ResponseProduct> product = new_device_under_test();

  ASSERT_FALSE(product->has_optimal_fft());

  product->add_response(scalarA);

  ASSERT_FALSE(product->has_optimal_fft());

  product->add_response(scalarB);

  ASSERT_TRUE(product->has_optimal_fft());
  ASSERT_EQ(product->get_optimal_fft(), optimal);
}



} // namespace dsp::test
