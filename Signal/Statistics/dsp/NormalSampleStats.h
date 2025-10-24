/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_NormalSampleStats_h
#define __dsp_NormalSampleStats_h

#include <cmath>
#include <inttypes.h>

namespace dsp
{
  /**
   * @brief Computes the stddev of the sample mean and variance for a normal distribution
   */
  class NormalSampleStats
  {
      uint64_t ndat = 0;    //! number of instances in sample
      double variance = 0;  //! variance of instances in sample

    public:

      //! Set the size of the sample
      void set_ndat (uint64_t n) { ndat = n; }

      //! Set the variance of the instances in the sample
      void set_variance (double var) { variance = var; }

      //! Return the standard deviation of the sample mean
      double get_sample_mean_stddev () const { return std::sqrt(variance / ndat); }

      //! Return the standard deviation of the sample variance
      /*! See equation 57 at https://mathworld.wolfram.com/NormalDistribution.html */
      double get_sample_variance_stddev () const { return std::sqrt(2.0 * (ndat - 1.0)) * variance / ndat; }
  };

} // namespace dsp

#endif // __dsp_NormalSampleStats_h
