//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/GenericSixteenBitUnpacker.h

#ifndef __GenericSixteenBitUnpacker_h
#define __GenericSixteenBitUnpacker_h

#include "dsp/SixteenBitUnpacker.h"
#include "dsp/BitTable.h"

namespace dsp {

  //! Simple 16-bit to float unpacker for generic 16-bit files
  class GenericSixteenBitUnpacker : public SixteenBitUnpacker
  {

  public:
    
    //! Constructor
    GenericSixteenBitUnpacker (BitTable::Type encoding = BitTable::Type::TwosComplement);

  protected:

    //! Return true if this unpacker can convert the Observation
    virtual bool matches (const Observation* observation);

    //! Override BitUnpacker::unpack
    virtual void unpack ();

  private:

    BitTable::Type encoding{BitTable::Type::TwosComplement};  

  };

}

#endif // !defined(__GenericSixteenBitUnpacker_h)

