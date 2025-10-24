/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Truncate.h"
#include "dsp/Observation.h"

#include "Error.h"

using namespace std;

//! Constructor
dsp::Truncate::Truncate () 
  : Transformation <TimeSeries,TimeSeries> ("Truncate", inplace)
{
}

//! Detect the input data
void dsp::Truncate::transformation () try
{
  if (input->get_ndat() == 0)
    return;

  MJD start_time = input->get_start_time();
  MJD end_time = input->get_end_time();

  if (start_time >= end_epoch)
  {
    if (Operation::verbose)
      cerr << endl << "dsp::Truncate::transformation end of data" << endl;

    throw Error(EndOfFile, "dsp::Truncate::transformation",
                "input start_time=" + start_time.printdays(13) + 
                " >= end_epoch=" + end_epoch.printdays(13));
  }

  if (end_time >= end_epoch)
  {
    double end_seconds = (end_epoch - start_time).in_seconds();
    int64_t end_ndat = end_seconds * input->get_rate();

    if (Operation::verbose)
      cerr << endl << "dsp::Truncate::transformation the end is nigh " << end_seconds << " seconds = " << end_ndat << " samples" << endl;

    output->set_ndat(end_ndat);
  }
}
catch (Error& error)
{
  throw error += "dsp::Truncate::transformation";
}
