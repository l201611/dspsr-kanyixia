/***************************************************************************
 *
 *   Copyright (C) 2012 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericSixteenBitUnpacker.h"
#include "dsp/BitTable.h"

using namespace std;

//! Constructor
dsp::GenericSixteenBitUnpacker::GenericSixteenBitUnpacker (BitTable::Type _encoding)
  : SixteenBitUnpacker (_encoding, "GenericSixteenBitUnpacker")
{
}

bool dsp::GenericSixteenBitUnpacker::matches (const Observation* observation)
{
  return observation->get_nbit() == 16;
}

void dsp::GenericSixteenBitUnpacker::unpack ()
{
  dsp::SixteenBitUnpacker::unpack ();
}

