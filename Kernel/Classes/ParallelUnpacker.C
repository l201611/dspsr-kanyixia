/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ParallelUnpacker.h"
#include "dsp/ParallelInput.h"
#include "dsp/MemoryHost.h"

#include "Error.h"

using namespace std;

//! Constructor
dsp::ParallelUnpacker::ParallelUnpacker (const char* name)
  : Transformation <ParallelBitSeries, TimeSeries> (name, outofplace)
{
  output_order = TimeSeries::OrderFPT;
}

dsp::ParallelUnpacker * dsp::ParallelUnpacker::clone() const
{
  throw Error (InvalidState, "dsp::ParallelUnpacker::clone", "Not implemented in " + get_name());
  return 0;
}

void dsp::ParallelUnpacker::prepare ()
{
  if (verbose)
    cerr << "dsp::ParallelUnpacker::prepare" << endl;

  // set the Observation information
  output->Observation::operator=(*input);

  // cerr << "dsp::ParallelUnpacker::prepare input=" << input.get() << " output=" << output.get() << endl;

  if (verbose)
    cerr << "dsp::ParallelUnpacker::prepare output start_time=" << output->get_start_time() << endl;
}

void dsp::ParallelUnpacker::reserve ()
{
  // set the Observation information, required if subsequent transformations operate in-place on the data dimensions
  output->Observation::operator=(*input);

  output->set_order (output_order);

  if (verbose)
    cerr << "dsp::ParallelUnpacker::reserve input ndat=" << input->get_ndat() << endl;

  // resize the output
  output->resize (input->get_ndat());
}

uint64_t dsp::ParallelUnpacker::bytes_storage() const
{
  uint64_t total_bytes = 0;
  for (auto& op: unpackers)
    total_bytes += op->bytes_storage();

  return total_bytes;
}

uint64_t dsp::ParallelUnpacker::bytes_scratch () const
{
  uint64_t max_bytes = 0;
  for (auto& op: unpackers)
    max_bytes = std::max(max_bytes,op->bytes_scratch());

  return max_bytes;
}

//! Return true if the unpacker support the specified output order
bool dsp::ParallelUnpacker::get_order_supported (TimeSeries::Order order) const
{
  // by default, only the current order is supported
  return order == output_order;
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::ParallelUnpacker::set_output_order (TimeSeries::Order order)
{
  if (order != output_order)
    throw Error (InvalidState, "dsp::ParallelUnpacker::set_output_order",
		 "unsupported output order");
}

//! Return true if the unpacker can operate on the specified device
bool dsp::ParallelUnpacker::get_device_supported (Memory* memory) const
{
  return dynamic_cast<MemoryHost*>(memory) != nullptr;
}

//! Set the device on which the unpacker will operate
void dsp::ParallelUnpacker::set_device (Memory* memory)
{
  if (!get_device_supported (memory))
    throw Error (InvalidState, "dsp::ParallelUnpacker::set_device", "unsupported device memory");
}

//! Initialize and resize the output before calling unpack
void dsp::ParallelUnpacker::transformation ()
{
  if (verbose)
    cerr << "dsp::ParallelUnpacker::transformation" << endl;;

  // set the Observation information
  prepare ();

  reserve ();

  // unpack the data
  unpack ();

  if (verbose)
    cerr << "dsp::ParallelUnpacker::transformation TimeSeries book-keeping\n"
      "  input_sample=" << input->get_input_sample() <<
      "  seek=" << input->get_request_offset() <<
      "  ndat=" << input->get_request_ndat() << endl;;

  if (output->get_ndat())
  {
    // Set the input_sample attribute
    output->set_input_sample(input->get_input_sample());

    // The following lines deal with time sample resolution of the data source
    output->seek (input->get_request_offset());

    output->decrease_ndat (input->get_request_ndat());
  }

  if (verbose)
    cerr << "dsp::ParallelUnpacker::transformation exit" << endl;;
}

void dsp::ParallelUnpacker::set_buffering_policy (BufferingPolicy* policy)
{
  for (auto instance: unpackers)
    instance->set_buffering_policy (policy);
}

void dsp::ParallelUnpacker::set_cerr (std::ostream& os) const
{
  for (auto instance: unpackers)
    instance->set_cerr (os);
}

void dsp::ParallelUnpacker::match_resolution (ParallelInput* input)
{
  if (unpackers.size())
    unpackers.at(0)->match_resolution(input->at(0));
}

unsigned dsp::ParallelUnpacker::get_resolution () const
{
  if (unpackers.size())
    return unpackers.at(0)->get_resolution();
  else
    return 0;
}
