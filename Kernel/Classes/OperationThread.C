/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/OperationThread.h"
#include <errno.h>
#include <algorithm>

using namespace std;

dsp::OperationThread::OperationThread (Operation* op)
  : Operation ( "OperationThread" )
{
  if (op)
    operations.push_back( op );

  context = new ThreadContext;
  state = Idle;

  errno = pthread_create (&id, 0, operation_thread, this);

  if (errno != 0)
    throw Error (FailedSys, "dsp::OperationThread", "pthread_create");
}

dsp::OperationThread::~OperationThread ()
{
  if (verbose)
    cerr << "dsp::OperationThread::~OperationThread()" << endl;
}

void* dsp::OperationThread::operation_thread (void* ptr)
{
  reinterpret_cast<OperationThread*>( ptr )->thread ();
  return 0;
}

void dsp::OperationThread::thread () try
{
  ThreadContext::Lock lock (context);

  while (state != Quit)
  {
    while (state == Idle)
      context->wait ();

    if (state == Quit)
      return;

    for_each( operations.begin(), operations.end(),
	      mem_fun(&Operation::operate) );

    state = Idle;
    context->broadcast ();
  }
}
catch (Error& error)
{
  cerr << "dsp::OperationThread error" << error << endl;
  throw error += "dsp::OperationThread::thread";
}

void dsp::OperationThread::operation ()
{
  ThreadContext::Lock lock (context);
  while (state != Idle)
    context->wait ();

  state = Active;
  context->broadcast ();
}

void dsp::OperationThread::append_operation (Operation* op)
{
  if (state != Idle)
    throw Error (InvalidState, "dsp::OperationThread::append_operation",
       "cannot append operation when state != Idle");
  else
    if (op)
      operations.push_back( op );    
}

void dsp::OperationThread::prepare ()
{
  for_each( operations.begin(), operations.end(),
	    mem_fun(&Operation::prepare) );
}

void dsp::OperationThread::reserve ()
{
  for_each( operations.begin(), operations.end(),
	    mem_fun(&Operation::reserve) );
}

void dsp::OperationThread::add_extensions (Extensions* ext)
{
  for_each( operations.begin(), operations.end(),
	    bind2nd( mem_fun(&Operation::add_extensions), ext) );
}

void dsp::OperationThread::combine (const Operation* op)
{
  const OperationThread* top = dynamic_cast<const OperationThread*>( op );
  if (top)
  {
    if (top->operations.size() != operations.size())
      throw Error (InvalidState, "dsp::OperationThread::combine",
		   "thread length=%u != other thread length=%u",
		   operations.size(), top->operations.size());

    for (unsigned i=0; i<operations.size(); i++)
      operations[i] -> combine( top->operations[i] );
  }
  else if (operations.size() == 1)
    operations[0] -> combine( op );
  else
    throw Error (InvalidState, "dsp::OperationThread::combine",
		 "cannot combine single Operation when thread length=%u",
		 operations.size());
}

void dsp::OperationThread::report () const
{
  for_each( operations.begin(), operations.end(),
	    mem_fun(&Operation::report) );
}

void dsp::OperationThread::reset ()
{
  for_each( operations.begin(), operations.end(),
	    mem_fun(&Operation::reset) );
}

uint64_t dsp::OperationThread::bytes_storage() const
{
  uint64_t total_bytes = 0;
  for (auto& op: operations)
    total_bytes += op->bytes_storage();

  return total_bytes;
}

uint64_t dsp::OperationThread::bytes_scratch () const
{
  uint64_t max_bytes = 0;
  for (auto& op: operations)
    max_bytes = std::max(max_bytes,op->bytes_scratch());

  return max_bytes;
}

dsp::OperationThread::Wait::Wait (OperationThread* opthread)
  : Operation ( "OperationThread::Wait" )
{
  parent = opthread;
}

void dsp::OperationThread::Wait::operation ()
{
  ThreadContext::Lock lock (parent->context);

  while (parent->state == Active)
    parent->context->wait ();
}

dsp::OperationThread::Wait* dsp::OperationThread::get_wait()
{
  return new Wait(this);
}

