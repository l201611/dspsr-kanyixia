//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/SixteenBitUnpacker.h

#ifndef __SixteenBitUnpacker_h
#define __SixteenBitUnpacker_h

#include "dsp/HistUnpacker.h"
#include "dsp/BitTable.h"

namespace dsp {

  //! Converts 16-bit digitised samples to floating point
  class SixteenBitUnpacker: public HistUnpacker
  {
  public:

    //! Null constructor
    SixteenBitUnpacker (BitTable::Type encoding, const char* name = "SixteenBitUnpacker");

  protected:

    void unpack ();

    BitTable::Type encoding;

  private:

    void unpack_bits (uint64_t ndat, const int16_t * from, const unsigned nskip, float* into, const unsigned fskip, unsigned long* hist);

  };
}

#endif
