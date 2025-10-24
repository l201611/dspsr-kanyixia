/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*! \file ParallelInput_registry.C
  \brief Register dsp::ParallelInput-derived classes for use in this file
    
  Classes that inherit dsp::ParallelInput may be registered for use by
  utilizing the Registry::List<dsp::ParallelInput>::Enter<Type> template class.
  Static instances of this template class should be given a unique
  name and enclosed within preprocessor directives that make the
  instantiation optional.  There are plenty of examples in the source code.

  \note Do not change the order in which registry entries are made
  without testing all of the file types.  Ensure that anything
  added performs a proper is_valid() test.
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/*! SingleFile is built in */
#include "dsp/SingleFile.h"
static dsp::ParallelInput::Register::Enter<dsp::SingleFile> single_file;

/*! DualFile is built in */
#include "dsp/DualFile.h"
static dsp::ParallelInput::Register::Enter<dsp::DualFile> dual_file;

dsp::ParallelInput::Register& dsp::ParallelInput::get_register ()
{
  return Register::get_registry();
}

dsp::ParallelInput* dsp::ParallelInput::create (const std::string& descriptor)
{
  if (verbose)
    std::cerr << "dsp::ParallelInput::create descriptor='" << descriptor << std::endl;

  ParallelInput::Register& registry = get_register();

  if (verbose)
    std::cerr << "dsp::ParallelInput::create with " << registry.size() << " registered sub-classes" << std::endl;

  for (unsigned ichild=0; ichild < registry.size(); ichild++) try
  {
    if (verbose)
      std::cerr << "dsp::ParallelInput::create testing " << registry[ichild]->get_name() << std::endl;;

    if ( registry[ichild]->matches (descriptor) )
    {
      if (verbose)
        std::cerr << "dsp::ParallelInput::create " << registry[ichild]->get_name()
                  << "::matches() returned true" << std::endl;

      ParallelInput* child = registry.create (ichild);
      child->open( descriptor );
      return child;
    }
  }
  catch (Error& error)
  {
    if (verbose)
      std::cerr << "dsp::ParallelInput::create failed while testing "
                << registry[ichild]->get_name() << std::endl
                << error.get_message() << std::endl;
  }

  std::string msg = descriptor;

  msg += " not a recognized parallel input format\n\t"
      + tostring(registry.size()) + " registered Formats: ";

  for (unsigned ichild=0; ichild < registry.size(); ichild++)
    msg += registry[ichild]->get_name() + " ";

  throw Error (InvalidParam, "dsp::ParallelInput::create", msg);
}
