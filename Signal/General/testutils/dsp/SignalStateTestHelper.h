/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_SignalStateTestHelper_h
#define __dsp_SignalStateTestHelper_h

#include "Types.h"

#include <vector>

namespace dsp::test {

  /**
   * @brief Return a list of the valid number of polarisations for the specified signal state
   *
   * @param state polarisation state of the signal
   * @return std::vector<unsigned> list of valid numbers of polarisations for the state
   */
  std::vector<unsigned> get_npols_for_state(Signal::State state)
  {
    std::vector<unsigned> npols;
    if (state == Signal::Nyquist || state == Signal::Analytic)
    {
      npols.push_back(1);
      npols.push_back(2);
    }
    else if (state == Signal::Intensity)
    {
      npols.push_back(1);
    }
    else if (state == Signal::PPQQ)
    {
      npols.push_back(2);
    }
    else if (state == Signal::Stokes || state == Signal::Coherence)
    {
      npols.push_back(4);
    }
    return npols;
  }

  /**
   * @brief Return the number of dimensions for the specified polarisation signal state
   *
   * @param state polarisation state of the signal
   * @return unsigned number of dimensions for the state
   */
  unsigned get_ndim_for_state(Signal::State state)
  {
    if (state == Signal::Analytic)
      return 2;
    else
      return 1;
  }
} // namespace dsp::test

#endif // __dsp_SignalStateTestHelper_h
