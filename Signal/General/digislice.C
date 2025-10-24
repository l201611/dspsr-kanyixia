/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  digislice - slices timeseries into output floating-point DADA files
 */

#include "dsp/LoadToSlice.h"
#include "dsp/LoadToSliceN.h"
#include "dsp/MaskTimes.h"

#include "CommandLine.h"

#include <cassert>
using namespace std;

// The LoadToSlice configuration parameters
Reference::To<dsp::LoadToSlice::Config> config;

// The name of the file from which slices will be read
std::string slice_file;

void parse_options (int argc, char** argv) try
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;
  
  menu.set_help_header ("digislice - convert input to floating-point DADA file segments");
  menu.set_version ("digislice " + tostring(dsp::version) +
		    " <" + FTransform::get_library() + ">");

  config->add_options (menu);

  arg = menu.add (slice_file, 's', "file");
  arg->set_help ("slice filename");
  arg->set_long_help
    ("the slice file is an ASCII file with two columns: start_time (MJD) and duration (seconds) \n"
     "the first line in this file must be '# MJD duration'");

  arg = menu.add (config->output_filename, 'o', "file");
  arg->set_help ("output filename");

  bool revert = false;
  arg = menu.add (revert, 'p');
  arg->set_help ("revert to FPT order");

  menu.parse (argc, argv);
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
  config = new dsp::LoadToSlice::Config;
  config->application_name = "digislice";
  config->dedisp.dispersion_measure = 0.0;

  parse_options (argc, argv);

  if (slice_file.empty())
  {
    cerr << "digislice: please specify slice file with -s option" << endl;
    return -1;
  }

  Reference::To<dsp::MaskTimes> slices = new dsp::MaskTimes;
  slices->load(slice_file);

  if (slices->size() == 0)
  {
    cerr << "digislice: no slices loaded from " << slice_file << endl;
    return -1;
  }

  Reference::To<dsp::Pipeline> engine;

  if (config->get_total_nthread() > 1)
    engine = new dsp::LoadToSliceN (config);
  else
    engine = new dsp::LoadToSlice (config);

  auto slicer = dynamic_cast<dsp::CanSliceTime*>(engine.ptr());
  assert (slicer != nullptr);

  engine->set_source( config->open (argc, argv) );
  engine->construct ();
  engine->prepare ();

  for (unsigned islice=0; islice < slices->size(); islice++)
  {
    auto interval = slices->get_interval(islice);
    cerr << "digislice: islice=" << islice << " start=" << interval.start_time.printdays(13) << endl;

    slicer->set_start_time(interval.start_time - interval.duration);
    slicer->set_end_time(interval.end_time);

    engine->run();
    engine->finish();
  }
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}
