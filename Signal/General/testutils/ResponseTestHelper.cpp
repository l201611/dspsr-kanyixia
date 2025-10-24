/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ResponseTestHelper.h"
#include <gtest/gtest.h>

namespace dsp::test
{
  void ResponseTestHelper::configure(const Observation*, unsigned)
  {
    if (ndat == 0)
      ndat = 1024;
  }

  void ResponseTestHelper::build(const Observation*)
  {
    // disable setting all values to zero in Response::build, 
    // and count the number of times called
    build_counter ++;
  }

//
// TEST HELPERS
//

// fill a vector of complex<float> with a pure tone
void create_test_vector(std::vector<std::complex<float>>& phasors)
{
  unsigned ndat = phasors.size();
  for (unsigned idat=0; idat < ndat; idat++)
  {
    double phase = 2.0 * M_PI * double(idat) / double(ndat);
    phasors[idat] = std::complex<float> (cos(phase), sin(phase));
  }
}

// verify that response contains test vector
void verify_equality(const std::vector<std::complex<float>>& phasors, const dsp::Response* response)
{
  unsigned ndat = phasors.size();
  constexpr unsigned ndim = 2;

  ASSERT_EQ(response->get_ndim(), ndim);  // Re,Im
  ASSERT_EQ(response->get_nchan(), 1);    // single channel
  ASSERT_EQ(response->get_npol(), 1);     // single response transforms both polarizations
  ASSERT_EQ(response->get_ndat(), ndat);  // number of samples in frequency response

  // access data via Shape::get_datptr (unsigned ichan, unsigned ipol)
  auto data = response->get_datptr(0,0);

  for (unsigned idat=0; idat < ndat; idat++)
  {
    auto real = data[idat*ndim];
    auto imag = data[idat*ndim + 1];

    EXPECT_FLOAT_EQ(real, phasors[idat].real()) << " idat=" << idat << std::endl;
    EXPECT_FLOAT_EQ(imag, phasors[idat].imag()) << " idat=" << idat << std::endl;
  }
}

// verify that two vectors are identical
void verify_equality(const std::vector<std::complex<float>>& A, const std::vector<std::complex<float>>& B)
{
  ASSERT_EQ(B.size(), A.size());

  unsigned ndat = A.size();
  for (unsigned idat=0; idat < ndat; idat++)
  {
    EXPECT_FLOAT_EQ(A[idat].real(), B[idat].real()) << " idat=" << idat << std::endl;
    EXPECT_FLOAT_EQ(A[idat].imag(), B[idat].imag()) << " idat=" << idat << std::endl;
  }
}

// fill a vector of Jones<float> with unique pure tones in each element
void create_test_vector(std::vector<Jones<float>>& matrices)
{
  unsigned ndat = matrices.size();
  for (unsigned idat=0; idat < ndat; idat++)
  {
    double phase = 2.0 * M_PI * double(idat) / double(ndat);
    double cosphi = cos(phase);
    double sinphi = sin(phase);
    std::complex<float> j00 (cosphi, sinphi);
    std::complex<float> j01 (sinphi, cosphi);
    std::complex<float> j10 (cosphi, -sinphi);
    std::complex<float> j11 (-sinphi, cosphi);

    matrices[idat] = Jones<float> (j00, j01, j10, j11);
  }
}

// verify that response contains test vector
void verify_equality(const std::vector<Jones<float>>& matrices, const dsp::Response* response)
{
  unsigned ndat = matrices.size();

  // number of components in 2x2 matrix
  unsigned ncomp = 4;

  ASSERT_EQ(response->get_ndim(),  2*ncomp); // Re,Im for each complex-valued component
  ASSERT_EQ(response->get_nchan(), 1);       // single channel
  ASSERT_EQ(response->get_npol(),  1);       // single response transforms both polarizations
  ASSERT_EQ(response->get_ndat(), ndat);     // number of samples in frequency response

  // access data via Shape::get_datptr (unsigned ichan, unsigned ipol)
  auto data = response->get_datptr(0,0);

  // in the Response class, the elements of a Jones matrix are ordered as: j00, j10, j11, j01
  unsigned transposed[4] = { 0, 2, 3, 1 };

  for (unsigned idat=0; idat < ndat; idat++)
  {
    for (unsigned icomp=0; icomp < ncomp; icomp++)
    {
      auto c = matrices[idat][transposed[icomp]];

      auto real = *data; data ++;
      auto imag = *data; data ++;

      EXPECT_FLOAT_EQ(real, c.real()) << " idat=" << idat << " icomp=" << icomp << std::endl;
      EXPECT_FLOAT_EQ(imag, c.imag()) << " idat=" << idat << " icomp=" << icomp << std::endl;
    }
  }
}

} // namespace dsp::test
