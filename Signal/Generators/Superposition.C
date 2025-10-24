/**************************************************************************2
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Superposition.h"

using namespace std;

dsp::Superposition::Superposition () : Generator("Superposition")
{
  source_output = new TimeSeries;
}

void dsp::Superposition::operation ()
{
  if (verbose)
    cerr << "dsp::Superposition::operation" << endl;

  output->copy_configuration(get_info());
  output->resize(block_size);
  output->set_input_sample(current_sample);

  if (block_size == 0)
    return;

  output->zero();

  if (sources.size() == 0)
  {
    if (verbose)
      cerr << "dsp::Superposition::operation no sources" << endl;
    return;
  }

  for (unsigned isource=0; isource < sources.size(); isource++)
  {
    sources[isource]->get_info()->copy(get_info());
    sources[isource]->set_output(source_output);
    sources[isource]->set_block_size(block_size);
    sources[isource]->operate();
    output->add(source_output);
  }

  current_sample += block_size;

  if (verbose)
    cerr << "dsp::Superposition::operation done" << endl;
}

void dsp::Superposition::add (Source* source)
{
  sources.push_back(source);

  if (sources.size() == 1)
  {
    // Copy the observation attributes of the first source added
    get_info()->copy(source->get_info());
  }
}