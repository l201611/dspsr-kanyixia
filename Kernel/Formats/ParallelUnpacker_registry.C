/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*! \file ParallelUnpacker_registry.C
  \brief Register dsp::ParallelUnpacker-derived classes for use in this file
    
  Classes that inherit dsp::ParallelUnpacker may be registered for use by
  utilizing the Registry::List<dsp::ParallelUnpacker>::Enter<Type> template class.
  Static instances of this template class should be given a unique
  name and enclosed within preprocessor directives that make the
  instantiation optional.  There are plenty of examples in the source code.

  \note Do not change the order in which registry entries are made
  without testing all of the file types.  Ensure that anything
  added performs a proper matches() test.
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#if HAVE_ska1
#include "dsp/SKAParallelUnpacker.h"
static dsp::ParallelUnpacker::Register::Enter<dsp::SKAParallelUnpacker> ska;
#endif

/*! SingleUnpacker is built in */
#include "dsp/SingleUnpacker.h"
static dsp::ParallelUnpacker::Register::Enter<dsp::SingleUnpacker> single_file;

dsp::ParallelUnpacker::Register& dsp::ParallelUnpacker::get_register ()
{
  return Register::get_registry();
}

dsp::ParallelUnpacker* dsp::ParallelUnpacker::create (const Observation* observation) try
{
  Register& registry = get_register();

  if (verbose)
    std::cerr << "dsp::ParallelUnpacker::create with " << registry.size()
              << " registered sub-classes" << std::endl;

  for (unsigned ichild=0; ichild < registry.size(); ichild++)
  {
    if (verbose)
      std::cerr << "dsp::ParallelUnpacker::create testing "
                << registry[ichild]->get_name() << std::endl;

    if ( registry[ichild]->matches (observation) )
    {
      auto child = registry.create (ichild);
      child-> match( observation );

      if (verbose)
        std::cerr << "dsp::ParallelUnpacker::create return new sub-class" << std::endl;

      return child;

    }
  }

  throw Error (InvalidState, std::string(),
               "no unpacker for machine=" + observation->get_machine());
}
catch (Error& error)
{
  throw error += "dsp::ParallelUnpacker::create";
}

