/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ParallelBitSeries.h"

using namespace std;

dsp::ParallelBitSeries::ParallelBitSeries()
{

}

//! Create and manage N BitSeries objects
void dsp::ParallelBitSeries::resize (unsigned N)
{
  if (verbose)
    cerr << "dsp::ParallelBitSeries::resize N=" << N << endl;

  bitseries.resize(N);
  for (auto& instance: bitseries)
  {
    if (!instance)
      instance = new BitSeries;
    if (memory)
      instance->set_memory(memory);
  }

  if (verbose)
    cerr << "dsp::ParallelBitSeries::resize done" << endl;
}

void dsp::ParallelBitSeries::set_memory (Memory* _memory)
{
  memory = _memory;
  for (auto& instance: bitseries)
    instance->set_memory(memory);
}

unsigned dsp::ParallelBitSeries::get_request_offset () const
{
  return bitseries.at(0)->get_request_offset();
}

uint64_t dsp::ParallelBitSeries::get_request_ndat () const
{
  return bitseries.at(0)->get_request_ndat();
}

int64_t dsp::ParallelBitSeries::get_input_sample (Input* input) const
{
  return bitseries.at(0)->get_input_sample(input);
}

void dsp::ParallelBitSeries::copy_configuration (const Observation* copy)
{
  if (copy == this)
    return;

  if (verbose)
    cerr << "dsp::ParallelBitSeries::copy_configuration this=" << this << " copy=" << copy << endl;

  Observation::operator=( *copy );

  if (verbose)
    cerr << "dsp::ParallelBitSeries::copy_configuration ndat=" << get_ndat() << endl;
}
