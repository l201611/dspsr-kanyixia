//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004-2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/FourBitUnpacker.h

#ifndef __FourBitUnpacker_h
#define __FourBitUnpacker_h

#include "dsp/BitUnpacker.h"

namespace dsp {

  //! Converts 4-bit digitised samples to floating point
  class FourBitUnpacker: public BitUnpacker
  {

  public:

    //! Null constructor
    FourBitUnpacker (const char* name = "FourBitUnpacker");

    //! Get the histogram for the specified digitizer
    void get_histogram (std::vector<unsigned long>&, unsigned idig) const;

  protected:

    //! Unpack a single real-valued process (e.g. single digitizer output)
    /*!
    Input bytes are interleaved in time, frequency, polarization, dimension (TFPD) order.
    Each byte contains two consecutive time samples from a single real-valued process.

    @param ndat total number of real-valued time samples
    @param from base address of TFPD-interleaved digitized samples
    @param from_byte_stride byte offset between consecutive pairs of digitized sample(s)
    @param into base address of unpacked time-contiguous floating point data
    @param into_float_stride offset to next consecutive real-valued sample
    @param hist base address of histogram of digitized states
    */
    void unpack_bits
    (
      uint64_t ndat,
      const unsigned char* from,
      const unsigned from_byte_stride,
      float* into,
      const unsigned into_float_stride,
      unsigned long* hist
    ) override;

  };
}
#endif
