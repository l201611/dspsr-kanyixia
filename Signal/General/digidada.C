/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  digidada - requantizes input and writes out a single TFP-order DADA file
 */

#include "dsp/dsp.h"
#include "dsp/LoadToQuantize.h"
#include "dsp/DADAOutputFile.h"

#include "dsp/ASCIIObservation.h"
#include "dsp/Source.h"

#if WE_DECIDE_TO_USE_MULTIPLE_THREADS
#include "dsp/LoadToQuantizeN.h"
#endif

#include "CommandLine.h"

#include <cassert>
using namespace std;

// The LoadToQuantize configuration parameters
Reference::To<dsp::LoadToQuantize::Config> config;

string parse_options (int argc, char** argv) try
{
  CommandLine::Menu menu;

  menu.set_help_header ("digidada - requantize input and output DADA file");
  menu.set_version ("digidada " + tostring(dsp::version));

  config->add_options (menu);

  string output_file;
  auto arg = menu.add(output_file, 'o', "output_file");
  arg->set_help("path to the output DADA file to which requantised data will be written");

  menu.parse (argc, argv);

  return output_file;
}
catch (Error& error)
{
  cerr << error << endl;
  exit (-1);
}
catch (std::exception& error)
{
  cerr << error.what() << endl;
  exit (-1);
}

int main (int argc, char** argv) try
{
  config = new dsp::LoadToQuantize::Config;
  config->application_name = "digidada";

  string output_file = parse_options (argc, argv);

  Reference::To<dsp::Pipeline> engine;
  dsp::DADAOutputFile sink(output_file.c_str());

#if WE_DECIDE_TO_USE_MULTIPLE_THREADS
  if (config->get_total_nthread() > 1)
    engine = new dsp::LoadToQuantizeN(&sink, config);
  else
#endif
    engine = new dsp::LoadToQuantize(&sink, config);

  auto source = config->open(argc, argv);
  auto dada_header = dynamic_cast<dsp::ASCIIObservation*>(source->get_info());

  bool verbose = dsp::Observation::verbose;

  if (dada_header)
  {
    if (verbose)
      cerr << "digidada: copying DADA header from source to DADA output file" << endl;
    sink.get_header()->set_header(dada_header->get_header());
  }

  engine->set_source(source);
  engine->construct();
  engine->prepare();
  engine->run();
  engine->finish();

  return 0;
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}
