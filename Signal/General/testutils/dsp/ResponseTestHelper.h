/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Response.h"

#ifndef __dsp_ResponseTestHelper_h
#define __dsp_ResponseTestHelper_h

namespace dsp::test
{
  class ResponseTestHelper : public Response
  {
  protected:
    unsigned build_counter = 0;
    void build(const Observation*) override;
    void configure(const Observation*, unsigned channels = 0) override;

  public:
    unsigned get_build_counter() const { return build_counter; }
  };

  // TEST HELPERS

  // fill a vector of complex<float> with a pure tone
  void create_test_vector(std::vector<std::complex<float>>& phasors);

  // verify that response contains test vector
  void verify_equality(const std::vector<std::complex<float>>& phasors, const dsp::Response* response);

  // verify that two vectors are identical
  void verify_equality(const std::vector<std::complex<float>>& A, const std::vector<std::complex<float>>& B);

  // fill a vector of Jones<float> with unique pure tones in each element
  void create_test_vector(std::vector<Jones<float>>& matrices);

  // verify that response contains test vector
  void verify_equality(const std::vector<Jones<float>>& matrices, const dsp::Response* response);

} // namespace dsp::test

#endif // __dsp_ResponseTestHelper_h
