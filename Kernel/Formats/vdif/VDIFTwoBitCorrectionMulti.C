/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/VDIFTwoBitCorrectionMulti.h"
#include "dsp/TwoBitTable.h"

bool dsp::VDIFTwoBitCorrectionMulti::matches (const Observation* observation)
{
  return observation->get_machine() == "VDIF" 
      && observation->get_nbit() == 2
      && (observation->get_npol() > 1 || observation->get_nchan() > 1);
}

//! Null constructor
dsp::VDIFTwoBitCorrectionMulti::VDIFTwoBitCorrectionMulti ()
  : SubByteTwoBitCorrection ("VDIFTwoBitCorrectionMulti")
{
  table = new TwoBitTable (TwoBitTable::OffsetBinary);
  // table = new TwoBitTable (TwoBitTable::TwosComplement);
}

void dsp::VDIFTwoBitCorrectionMulti::unpack ()
{
  if (get_ndig() != input->get_nchan() * input->get_npol())
    set_ndig(input->get_nchan() * input->get_npol());

  SubByteTwoBitCorrection::unpack ();
}

