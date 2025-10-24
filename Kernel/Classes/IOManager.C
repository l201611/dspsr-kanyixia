/***************************************************************************
 *
 *   Copyright (C) 2002 - 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/IOManager.h"
#include "dsp/File.h"
#include "dsp/BitSeries.h"
#include "dsp/Unpacker.h"
#include "dsp/TimeSeries.h"

#include "templates.h"
#include "Error.h"

using namespace std;

dsp::IOManager* dsp::IOManager::clone() const
{
  return new IOManager;
}

//! Prepare the appropriate Input and Unpacker
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
void dsp::IOManager::open (const string& id) try
{
  set_input ( File::create(id) );
}
catch (Error& error)
{
  throw error += "dsp::IOManager::open";
}
