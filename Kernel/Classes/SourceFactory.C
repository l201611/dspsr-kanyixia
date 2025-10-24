/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SourceFactory.h"

#include "dsp/ParallelInput.h"
#include "dsp/ParallelIOManager.h"

#include "dsp/File.h"
#include "dsp/IOManager.h"

#include "Error.h"

using namespace std;

//! Return a pointer to a new instance of the appropriate sub-class
dsp::Source* dsp::SourceFactory::create (const std::string& descriptor)
{
  if (Operation::verbose)
    std::cerr << "dsp::SourceFactory::create filename='" << descriptor << endl;

  string error_messages;

  try
  {
    auto file = File::create (descriptor.c_str());

    if (Operation::verbose)
      std::cerr << "dsp::SourceFactory::create File " << file->get_name() << " created" << endl;

    auto manager = new IOManager;
    manager->set_input(file);
    return manager;
  }
  catch (Error& error)
  {
    error_messages += error.get_message();
  }

  try
  {
    auto input = ParallelInput::create (descriptor.c_str());

    if (Operation::verbose)
      std::cerr << "dsp::SourceFactory::create ParallelInput " << input->get_name() << " created" << endl;

    auto manager = new ParallelIOManager;
    manager->set_input(input);
    return manager;
  }
  catch (Error& error)
  {
    cerr << "dsp::SourceFactory::create ParallelInput error=" << error << endl;
    if (error_messages.length())
      error_messages += "\n";

    error_messages += error.get_message();
  }

  throw Error (InvalidParam, "dsp::SourceFactory::create", error_messages);
}