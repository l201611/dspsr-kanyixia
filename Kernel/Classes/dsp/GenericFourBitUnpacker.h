//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2017-2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __GenericFourBitUnpacker_h
#define __GenericFourBitUnpacker_h

#include "dsp/FourBitUnpacker.h"

namespace dsp
{
  //! Converts data from 4-bit TFPD nibbles to 32-bit FPTD floating point values
  /*! This unpacker assumes that each byte contains two values in a time,
    frequency, polarization, dimension (TFPD) ordered sequence of 4-bit nibbles.

    For example, if ndim=2, then the real and imaginary parts of a single 
    time sample are packed into one byte; if ndim=1 and npol=2, then the
    two components of a single time sample of the electric field are packed
    into on byte.
    */
  class GenericFourBitUnpacker: public FourBitUnpacker
  {
  public:

    //! Constructor initializes bit table
    GenericFourBitUnpacker ();

    //! Return true if this unpacker can handle the observation
    bool matches (const Observation*);

  protected:

    //! Unpack 4-bit nibbles that are interleaved in TFPD order
    void unpack () override;

    //! Unpack two floats at a time from each byte
    /*!
    Input nibbles are interleaved in time, frequency, polarization, dimension (TFPD) order.

    @param ndat total number of floats to unpack
    @param from base address of TFPD-interleaved digitized samples
    @param from_byte_stride byte offset between pairs of digitized nibbles
    @param into base address of unpacked floating point data
    @param into_float_stride offset to next unpacked float
    @param hist base address of histogram of digitized states
    */
    void unpack_bits
    ( uint64_t ndat,
      const unsigned char* from,
      const unsigned nskip,
      float* into,
      const unsigned fskip,
      unsigned long* hist
    ) override;
  };
}

#endif
