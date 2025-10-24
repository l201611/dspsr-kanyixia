/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ParallelInput.h"
#include "dsp/ParallelBitSeries.h"
#include "dsp/Input.h"

#include "Error.h"

using namespace std;

dsp::ParallelInput::ParallelInput (const char* name) : Operation (name)
{
}

dsp::ParallelInput::~ParallelInput ()
{
}

const dsp::Input* dsp::ParallelInput::at (unsigned index) const { return inputs.at(index); }

dsp::Input* dsp::ParallelInput::at (unsigned index) { return inputs.at(index); }

dsp::Observation* dsp::ParallelInput::get_info () { return info; }

const dsp::Observation* dsp::ParallelInput::get_info () const { return info; }

void dsp::ParallelInput::set_output (ParallelBitSeries* _output)
{
  output = _output;

  output->resize(inputs.size());
  for (unsigned i=0; i<inputs.size(); i++)
  {
    inputs[i]->set_output(output->at(i));
  }
}

dsp::ParallelBitSeries* dsp::ParallelInput::get_output ()
{
  return output;
}

void dsp::ParallelInput::prepare ()
{
  if (verbose)
    cerr << "dsp::ParallelInput::prepare" << endl;

  // set the Observation information
  for (unsigned i=0; i<inputs.size(); i++)
  {
    get_output()->at(i)->copy_configuration(inputs.at(i)->get_info());
  }

  get_output()->copy_configuration(get_info());
  get_output()->set_ndat (0);

  if (verbose)
    cerr << "dsp::ParallelInput::prepare output start_time=" << output->get_start_time() << endl;
}

//! Reserve the maximum amount of output space required
void dsp::ParallelInput::reserve ()
{
  reserve (get_output());
}

void dsp::ParallelInput::reserve (ParallelBitSeries* bitseries)
{
  bitseries->resize(inputs.size());
  for (unsigned i=0; i<inputs.size(); i++)
  {
    inputs[i]->reserve(bitseries->at(i));
  }
}

void dsp::ParallelInput::combine (const Operation* other)
{
  Operation::combine (other);

  auto like = dynamic_cast<const ParallelInput*>( other );
  if (!like)
    return;

  for (unsigned i=0; i<inputs.size(); i++)
  {
    inputs[i]->combine(like->at(i));
  }
}

void dsp::ParallelInput::report () const
{
  for (auto instance: inputs)
    instance->report();
}

void dsp::ParallelInput::reset ()
{
  Operation::reset();
  for (auto instance: inputs)
    instance->reset();
}

void dsp::ParallelInput::reset_time () const
{
  Operation::reset_time();
  for (auto instance: inputs)
    instance->reset_time();
}

bool dsp::ParallelInput::eod() const
{
  if (verbose)
  {
    for (unsigned i=0; i < inputs.size(); i++)
      cerr << "dsp::ParallelInput::eod input["<< i <<"].eod=" << inputs[i]->eod() << endl;
  }

  for (auto instance: inputs)
    if (instance->eod())
      return true;

  if (verbose)
    cerr << "dsp::ParallelInput::eod return false" << endl;

  return false;
}

void dsp::ParallelInput::restart ()
{
  for (auto instance: inputs)
    instance->restart();
}

void dsp::ParallelInput::close ()
{
  for (auto instance: inputs)
    instance->close();
}

void dsp::ParallelInput::load (ParallelBitSeries* bitseries)
{
  if (verbose)
    cerr << "dsp::ParallelInput::load before lock" << endl;
  ThreadContext::Lock lock (context);
  if (verbose)
    cerr << "dsp::ParallelInput::load after lock" << endl;

  if (verbose)
    cerr << "dsp::ParallelInput::load" << endl;

  bitseries->resize(inputs.size());

  for (unsigned i=0; i<inputs.size(); i++)
  {
    Input* input = inputs.at(i);
    BitSeries* bits = bitseries->at(i);

    if (verbose)
      cerr << "dsp::ParallelInput::load input[" << i << "]=" << input
        << " bitseries=" << bits << endl;

    input->load(bits);
  }
}

void dsp::ParallelInput::seek (int64_t offset, int whence)
{
  double rate = inputs.at(0)->get_info()->get_rate();
  for (auto instance: inputs)
  {
    if (instance->get_info()->get_rate() != rate)
      throw Error (InvalidState, "dsp::ParallelInput::seek", "cannot seek to the same sample offset in every Input");

    instance->seek(offset,whence);
  }
}

void dsp::ParallelInput::seek(const MJD& mjd) try
{
  seek(get_info()->get_idat(mjd), SEEK_SET);
}
catch (Error& error)
{
  throw error += "dsp::ParallelInput::seek";
}

uint64_t dsp::ParallelInput::tell () const
{
  return inputs.at(0)->tell();
}

void dsp::ParallelInput::set_start_seconds (double seconds)
{
  for (auto instance: inputs)
    instance->set_start_seconds(seconds);
}

void dsp::ParallelInput::set_total_seconds (double seconds)
{
  for (auto instance: inputs)
    instance->set_total_seconds(seconds);
}

double dsp::ParallelInput::tell_seconds () const
{
  return inputs.at(0)->tell_seconds();
}

//! Load data into the ParallelBitSeries specified by set_output
void dsp::ParallelInput::operation ()
{
  load (get_output());
}

uint64_t dsp::ParallelInput::bytes_storage() const
{
  uint64_t total_bytes = 0;
  for (auto& op: inputs)
    total_bytes += op->bytes_storage();

  return total_bytes;
}

uint64_t dsp::ParallelInput::bytes_scratch () const
{
  uint64_t max_bytes = 0;
  for (auto& op: inputs)
    max_bytes = std::max(max_bytes,op->bytes_scratch());

  return max_bytes;
}
