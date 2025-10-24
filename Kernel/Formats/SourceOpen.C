/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SourceOpen.h"
#include "dsp/SourceFactory.h"

#include "dsp/IOManager.h"
#include "dsp/ParallelInput.h"
#include "dsp/ParallelUnpacker.h"
#include "dsp/SerialFiles.h"
#include "dsp/CommandLineHeader.h"

#include "dirutil.h"
#include <unistd.h>

using namespace std;

dsp::Source* dsp::SourceOpen::open (int argc, char** argv)
{
  vector<string> filenames;

  if (command_line_header)
  {
    CommandLineHeader clh;
    filenames.push_back ( clh.convert(argc,argv) );
  }

  else
  {
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);
  }

  unsigned nfile = filenames.size();

  if (nfile == 0)
  {
    cerr << "please specify filename[s] (or -h for help)" << endl;
    exit (-1);
  }

  if (nfile == 1)
  {
    if (Operation::verbose)
      cerr << "opening " << filenames[0] << endl;

    SourceFactory factory;
    return factory.create( filenames[0] );
  }

  if (Operation::verbose)
  {
    cerr << "opening contiguous data files: " << endl;
    for (unsigned ii=0; ii < filenames.size(); ii++)
      cerr << "  " << filenames[ii] << endl;
  }

  dsp::SerialFiles* multi = new dsp::SerialFiles;

  if (force_contiguity)
    multi->force_contiguity();

  multi->open (filenames);

  Reference::To<IOManager> manager = new IOManager;
  manager->set_input(multi);
  return manager.release();
}

template<class Container>
void print_names(const Container& container)
{
  for (auto& element: container)
    cout << element->get()->get_name() << endl;
}

void dsp::SourceOpen::list_backends()
{
  cout << endl << "available backend file formats" << endl;
  auto& files = File::get_register();
  print_names(files);
  auto& parallel_inputs = ParallelInput::get_register();
  print_names(parallel_inputs);

  cout << endl << "available backend interpreters" << endl;
  auto& unpackers = Unpacker::get_register();
  print_names(unpackers);
  auto& parallel_unpackers = ParallelUnpacker::get_register();
  print_names(parallel_unpackers);

  cout << endl;
}
