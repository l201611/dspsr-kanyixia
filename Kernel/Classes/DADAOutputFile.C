/***************************************************************************
 *
 *   Copyright (C) 2024 by Andrew Jameson & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADAOutputFile.h"
#include "dsp/ASCIIObservation.h"

//#include <fcntl.h>

#include <cstring>
using namespace std;

dsp::DADAOutputFile::DADAOutputFile(const char* filename)
  : OutputFile("DADAOutputFile")
{
  if (filename)
    output_filename = filename;
}

dsp::DADAOutputFile::DADAOutputFile(const std::string& filename)
  : OutputFile("DADAOutputFile")
{
  output_filename = filename;
}

std::string dsp::DADAOutputFile::get_extension() const
{
  return ".dada";
}

void dsp::DADAOutputFile::write_header ()
{
  ASCIIObservation ascii (get_input());
  ascii.set_machine ("dspsr");

  if (dada_header.size() == 0)
    dada_header.resize();

  ascii.unload(dada_header.get_header());
  unload_bytes(dada_header.get_header(), dada_header.size());
}

