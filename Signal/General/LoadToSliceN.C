/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToSliceN.h"

#include "dsp/OutputFile.h"
#include "dsp/OutputFileShare.h"
#include "FTransformAgent.h"
#include "ThreadContext.h"

#include <fstream>
#include <stdlib.h>
#include <errno.h>

using namespace std;

//! Constructor
dsp::LoadToSliceN::LoadToSliceN (LoadToSlice::Config* config)
{
  configuration = config;
  set_nthread (configuration->get_total_nthread());
}
    
//! Set the number of thread to be used
void dsp::LoadToSliceN::set_nthread (unsigned nthread)
{
  MultiThread::set_nthread (nthread);

  FTransform::nthread = nthread;

  if (configuration)
    set_configuration (configuration);
}

dsp::LoadToSlice* dsp::LoadToSliceN::at (unsigned i)
{
  return dynamic_cast<LoadToSlice*>( threads.at(i).get() );
}

//! Set the configuration to be used in prepare and run
void dsp::LoadToSliceN::set_configuration (LoadToSlice::Config* config)
{
  configuration = config;

  MultiThread::set_configuration (config);

  for (unsigned i=0; i<threads.size(); i++)
    at(i)->set_configuration( config );
}

void dsp::LoadToSliceN::share ()
{
  MultiThread::share ();

  if (at(0)->frequency_response && !at(0)->frequency_response->context)
    at(0)->frequency_response->context = new ThreadContext;

  // Output file sharing
  output_file = new OutputFileShare(threads.size());
  output_file->set_context(new ThreadContext);
  output_file->set_output_file(at(0)->outputFile);

  // Replace the normal output with shared version in each thread
  for (unsigned i=0; i<threads.size(); i++) 
  {
    at(i)->operations.pop_back(); // unload should be last....
    OutputFileShare::Submit* sub = output_file->new_Submit(i);
    sub->set_input(at(i)->outputFile->get_input());
    at(i)->operations.push_back(sub);
  }

}

//! The creator of new LoadToSlice threadss
dsp::LoadToSlice* dsp::LoadToSliceN::new_thread ()
{
  return new LoadToSlice;
}

//! Set the start of the time slice
void dsp::LoadToSliceN::set_start_time (const MJD& epoch)
{
  // only one thread needs to set the start epoch because this alters the shared input
  at(0)->seek_epoch(epoch);
}

//! Set the end of the time slice
void dsp::LoadToSliceN::set_end_time (const MJD& epoch)
{
  // all threads must set the end time because they each have their own Truncate operation
  for (unsigned i=0; i<threads.size(); i++) 
  {
    at(i)->set_end_time(epoch);
  }
}
