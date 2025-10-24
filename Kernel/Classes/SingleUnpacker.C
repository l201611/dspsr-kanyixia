/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SingleUnpacker.h"
#include "dsp/Unpacker.h"
#include "dsp/ParallelInput.h"

dsp::SingleUnpacker::SingleUnpacker () : ParallelUnpacker("SingleUnpacker") {}

dsp::SingleUnpacker::~SingleUnpacker () {}

bool dsp::SingleUnpacker::matches (const Observation* observation) const try
{
  Reference::To<Unpacker> unpacker = Unpacker::create(observation);
  return true;
}
catch (Error& error)
{
  return false;
}

void dsp::SingleUnpacker::match (const Observation* observation)
{
  Reference::To<Unpacker> unpacker = Unpacker::create(observation);
  unpackers.resize(1);
  unpackers[0] = unpacker;
}

void dsp::SingleUnpacker::unpack ()
{
  unpackers.at(0)->operate();
}

void dsp::SingleUnpacker::match_resolution (ParallelInput* input)
{
  unpackers.at(0)->match_resolution(input->at(0));
}

unsigned dsp::SingleUnpacker::get_resolution () const
{
  return unpackers.at(0)->get_resolution();
}
