/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/DADAFile.h"
#include "dsp/ASCIIObservation.h"
#include "dsp/BlockSize.h"

#include "ascii_header.h"
#include "FilePtr.h"
#include "Error.h"
#include "strutil.h"

#include <fstream>
#include <fcntl.h>

using namespace std;

dsp::DADAFile::DADAFile (const char* filename) : File ("DADA")
{
  if (filename)
    open (filename);
}

bool dsp::DADAFile::is_valid (const char* filename) const try
{
  DADAHeader tmp_hdr;
  tmp_hdr.load_header (filename);
  return tmp_hdr.valid(verbose);
}
catch (Error& error)
{
  if (verbose)
    cerr << "dsp::DADAFile::is_valid " << error.get_message() << endl;
  return false;
}

void dsp::DADAFile::open_file (const char* filename)
{
  if (verbose)
    cerr << "DADAFile::open filename=" << filename << endl;

  dada_header.load_header (filename);

  if (dada_header.empty())
    throw Error (FailedCall, "dsp::DADAFile::open_file", "get_header(%s) failed", filename);

  auto obs = new ASCIIObservation(dada_header.get_header());
  info = obs;

  header_bytes = dada_header.get_header_size();
  resolution = obs->get_resolution();

  if (verbose)
    cerr << "dsp::DADAFile::open_file header_bytes=" << header_bytes << endl;

  open_fd (filename);

  if (verbose)
    cerr << "DADAFile::open exit" << endl;
}
