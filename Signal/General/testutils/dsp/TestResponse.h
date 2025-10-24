/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Response.h"

#ifndef __dsp_test_TestResponse_h
#define __dsp_test_TestResponse_h

namespace dsp::test
{

//! A frequency response function with a known impulse response function
/*!
  The amplitude of the impulse response is described by a (possibly asymmetric
  Tukey window and the phase of the impulse response function within that window
  is described by one cycle of a complex sinusoid.  An additional half cycle
  of a sinusoid is added to both the real and imaginary parts to ensure that
  the mean of the impulse response is zero.  When testing real-valued input data,
  zero in the DC bin of the frequency response masks the fundamentally different
  nature of the spectrum.
*/
class TestResponse : public Response
{
  public:

    //! The temporal (impulse) response function
    std::vector<std::complex<float>> temporal;
    //! The spectral (frequency) response function
    std::vector<std::complex<float>> spectral;

    //! Configure everything required to define the response, but don't build
    /*!
      \pre Both impulse_neg and impulse_pos must be non-zero and ndim must be 2
    */
    void configure (const Observation* input, unsigned channels = 0) override;

    //! Build the response function
    void build (const Observation* input) override;
};

} // namespace dsp::test

#endif // __dsp_test_TestResponse_h
