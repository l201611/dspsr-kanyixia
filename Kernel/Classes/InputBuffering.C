/***************************************************************************
 *
 *   Copyright (C) 2005-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InputBuffering.h"
#include "dsp/Reserve.h"

using namespace std;

dsp::InputBuffering::InputBuffering (HasInput<TimeSeries>* _target)
{
  target = _target;
  next_start_sample = 0;

  name = "InputBuffering";
  reserve = new Reserve;
}

//! Set the target with input TimeSeries to be buffered
void dsp::InputBuffering::set_target (HasInput<TimeSeries>* _target)
{
  target = _target;
}

//! Set the maximum number of samples to be buffered
void dsp::InputBuffering::set_maximum_samples (uint64_t samples)
{
  reserve->reserve( get_input(), samples );
  resize_buffer();
}

void dsp::InputBuffering::resize_buffer ()
{
  if (!buffer)
  {
    if (Operation::verbose)
      cerr << "dsp::InputBuffering::resize_buffer null_clone input" << endl;
    buffer = get_input()->null_clone();
    if (Operation::verbose)
      buffer->set_cerr (cerr);
  }

  buffer->copy_dimensions( get_input() );

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::resize_buffer resize buffer"
      " reserved_samples=" << reserve->get_reserved() 
      << " (nchan=" << buffer->get_nchan() << " npol=" << buffer->get_npol() << ")" << endl;

  buffer->resize( reserve->get_reserved() );
}

uint64_t dsp::InputBuffering::bytes_storage() const
{
  if (!buffer)
    return 0;

  // the result is multiplied by 2 because space is also required for prepending data in the target input TimeSeries
  return buffer->internal_get_size() * 2;
}

/*! Copy remaining data from the target Transformation's input to buffer */
void dsp::InputBuffering::set_next_start (uint64_t next)
{
  const TimeSeries* input = get_input();

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start input=" << (void*) input << endl;

  next_start_sample = next;

  // the number of samples in the target
  const uint64_t ndat = input->get_ndat();

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start next=" << next 
         << " ndat=" << ndat 
         << " (nchan=" << input->get_nchan() << " npol=" << input->get_npol() << ")" << endl;

  if (ndat && input->get_input_sample() < 0)
    throw Error (InvalidState, "dsp::InputBuffering::set_next_start",
                 "input_sample of target input TimeSeries is not set");

  // the number of samples to be buffered
  uint64_t buffer_ndat = ndat - next_start_sample;

  if (next_start_sample > ndat)
    buffer_ndat = 0;

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start saving " << buffer_ndat << " samples" << endl;

  if ( buffer_ndat > reserve->get_reserved() )
  {
    if (Operation::verbose)
      cerr << "dsp::InputBuffering::set_next_start increasing reserve"
              " from " << reserve->get_reserved() << " to " << buffer_ndat << " samples" << endl;
    reserve->reserve( input, buffer_ndat );
  }

  // always resize the buffer to ensure the maximum dimensions are sufficient
  resize_buffer();

  buffer->set_ndat( buffer_ndat );

  if (buffer_ndat)
  {
    if (Operation::verbose)
      cerr << "dsp::InputBuffering::set_next_start copying from input sample "
          << input->get_input_sample() + next_start_sample << endl;

    buffer->copy_data( input, next_start_sample, buffer_ndat );
  }
  else
  {
    /* InputBuffering::Share::pre_transformation waits for get_next_contiguous 
       to return the desired sample index.  get_next_contiguous uses TimeSeries::input_sample,
       which is set by TimeSeries::copy_data when buffer_ndat > 0.
       TimeSeries::input_sample is set here to keep other threads from hanging when there
       are no samples in the buffer to be copied. */
    buffer->set_input_sample( input->get_input_sample() + next_start_sample );
  }

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::set_next_start resulting buffer"
         << " ndat=" << buffer->get_ndat() 
         << " nchan=" << buffer->get_nchan() << " npol=" << buffer->get_npol() << endl;
}

/*! Prepend buffered data to target Transformation's input TimeSeries */
void dsp::InputBuffering::pre_transformation () try
{
  if (!reserve->get_reserved() || !buffer || !buffer->get_ndat())
    return;

  const TimeSeries* container = get_input();

  int64_t want = container->get_input_sample();

  // don't wait for data preceding the first loaded block or last empty block
  if (want <= 0)
    return;

  if (buffer->get_input_sample() >= want)
  {
    if (Operation::verbose)
      cerr << "dsp::InputBuffering::pre_transformation buffer->input_sample=" << buffer->get_input_sample()
           << " > want=" << want << endl;
    return;
  }

  int64_t have = buffer->get_input_sample() + buffer->get_ndat();
  if (have > want)
  {
    if (Operation::verbose)
      cerr << "dsp::InputBuffering::pre_transformation buffer->get_ndat()"
           << buffer->get_ndat() << " have=" << have << " want=" << want << endl;
    buffer->set_ndat ( buffer->get_ndat() - have + want );
  }

  if (Operation::verbose)
  {
    cerr << "dsp::InputBuffering::pre_transformation prepend "
	       << buffer->get_ndat() << " samples"
         << " (nchan=" << buffer->get_nchan() << " npol=" << buffer->get_npol() << ")" << endl;
    cerr << "dsp::InputBuffering::pre_transformation target input sample="
         << want << endl;
  }

  const_cast<TimeSeries*>( container )->prepend (buffer);
}
catch (Error& error)
{
  throw error += "dsp::InputBuffering::pre_transformation";
}
/*! No action required after transformation */
void dsp::InputBuffering::post_transformation ()
{
}


int64_t dsp::InputBuffering::get_next_contiguous () const
{
  if (!buffer)
    return -1;

  return buffer->get_input_sample() + buffer->get_ndat();
}

