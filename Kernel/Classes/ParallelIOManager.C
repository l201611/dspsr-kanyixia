/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ParallelIOManager.h"
#include "dsp/ParallelInput.h"
#include "dsp/ParallelBitSeries.h"
#include "dsp/ParallelUnpacker.h"
#include "dsp/TimeSeries.h"

#include "templates.h"
#include "Error.h"

using namespace std;

dsp::ParallelIOManager* dsp::ParallelIOManager::clone() const
{
  return new ParallelIOManager;
}

//! Prepare the appropriate ParallelInput and ParallelUnpacker
/*!

  \param id string containing the id of the data source.  The source
  id may be a:
  <UL>
  <LI> filename
  <LI> a comma separated list of filenames to be treated as one observation
  <LI> a string of the form "IPC:xx", where "xx" is a shared memory key
  </UL>

  \pre This function is not fully implemented.
*/
void dsp::ParallelIOManager::open (const string& id) try
{
  set_input ( ParallelInput::create(id) );
}
catch (Error& error)
{
  throw error += "dsp::ParallelIOManager::open";
}
