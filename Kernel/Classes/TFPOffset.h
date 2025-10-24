//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024 by Jesmigel Cantos
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_TFPOffset_h
#define __dsp_TFPOffset_h

#include "dsp/TimeSeries.h"

#ifdef __host__
#define __prefix__ __host__ __device__
#else
#define __prefix__
#endif

namespace dsp {

  //! Computes array offset into TFP-ordered TimeSeries
  class TFPOffset
  {
    const unsigned dat_stride;
    const unsigned chan_stride;
    const unsigned pol_stride;

  public:

    TFPOffset(const TimeSeries* data) :
      dat_stride (data->get_nchan() * data->get_npol() * data->get_ndim()),
      chan_stride (data->get_npol() * data->get_ndim()),
      pol_stride (data->get_ndim())
    { }

    __prefix__ inline uint64_t operator() (uint64_t idat, unsigned ichan, unsigned ipol)
    {
      return idat * dat_stride + ichan * chan_stride + ipol * pol_stride;
    }

  };

} // namespace dsp

#endif // __dsp_TFPOffset_h
