/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADAHeader.h"
#include "dsp/dsp.h"
#include "ascii_header.h"
#include "FilePtr.h"
#include "Error.h"
#include "debug.h"
#include "strutil.h"

#include <cstring>
#include <cmath>

using namespace std;

const unsigned dsp::DADAHeader::default_header_size = 4096;

void dsp::DADAHeader::set_header(const char* ascii_header)
{
  if (!ascii_header)
    throw Error(InvalidState, "dsp::DADAHeader::set_header", "ascii_header == nullptr");

  unsigned header_size = 0;
  if (ascii_header_get(ascii_header, "HDR_SIZE", "%u", &header_size) < 0)
  {
    header_size = strlen(ascii_header);
    DEBUG("dsp::DADAHeader::set_header required HDR_SIZE=" << header_size);
  }

  resize(header_size);
  strncpy(get_header(), ascii_header, header_size);
}

void dsp::DADAHeader::resize(unsigned requested_header_size)
{
  unsigned header_size = default_header_size;
  while (header_size < requested_header_size)
    header_size *= 2;

  header.resize(header_size);
  memset(get_header(), 0, header_size);

  if (ascii_header_set(get_header(), "HDR_SIZE", "%d", header_size) < 0)
    throw Error (InvalidState, "dsp::DADAHeader::resize", "failed to set HDR_SIZE");
}

void dsp::DADAHeader::load_header (const char* filename)
{
  adjacent_header_file = false;

  FilePtr fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::DADAHeader::load_header", "fopen (%s)", filename);

  // default DADA header size
  hdr_size = 4096;
  char* c_header = 0;

  do
  {
    ::rewind (fptr);

    header.resize (hdr_size);
    c_header = header.data();

    if (fread (c_header, 1, hdr_size, fptr) != hdr_size)
      throw Error (FailedSys, "dsp::DADAHeader::load_header", "fread (nbyte=%u)", hdr_size);

    // ensure that text is null-terminated before calling ascii_header_get
    c_header[ hdr_size-1 ] = '\0';

    /* Get the header size */
    if (ascii_header_get (c_header, "HDR_SIZE", "%u", &hdr_size) != 1)
      hdr_size = 0;

    /* Ensure that the incoming header fits in the client header buffer */
  }
  while (hdr_size > header.size());

  if (hdr_size == 0)
  {
    // search for a matching .hdr file
    string hdr_ext = ".hdr";
    string hdr_fname = replace_extension (filename, hdr_ext);
    FilePtr hdr_ptr = fopen (hdr_fname.c_str(), "r");
    if (!fptr)
    {
      hdr_fname = filename + hdr_ext;
      hdr_ptr = fopen (hdr_fname.c_str(), "r");
    }

    if (!hdr_ptr)
      throw Error (InvalidState, "dsp::DADAHeader::load_header",
            "file has no header and no adjacent header file found");

    if (fseek (hdr_ptr, 0, SEEK_END) < 0)
      throw Error (FailedSys, "dsp::DADAHeader::load_header",
            "could not fseek to end of adjacent header file");

    hdr_size = ftell (hdr_ptr);
    if (hdr_size < 0)
      throw Error (FailedSys, "dsp::DADAHeader::load_header",
            "ftell fails at end of adjacent header file");

    ::rewind (hdr_ptr);

    header.resize (hdr_size);
    c_header = header.data();

    if (fread (c_header, 1, hdr_size, hdr_ptr) != hdr_size)
      throw Error (FailedSys, "dsp::DADAHeader::load_header",
            "fread (nbyte=%u) from header file", hdr_size);

    // ensure that text is null-terminated
    header[ hdr_size-1 ] = '\0';
    adjacent_header_file = true;

    // If set, HDR_SIZE defines a block of bytes to be skipped in the data file
    if (ascii_header_get(c_header, "HDR_SIZE", "%u", &hdr_size) != 1)
    {
      if (dsp::get_verbosity())
        cerr << "dsp::DADAHeader::load_header adjacent header file does not define HDR_SIZE" << endl;
      hdr_size = 0;
    }
    else
    {
      if (dsp::get_verbosity())
        cerr << "dsp::DADAHeader::load_header adjacent header file defines HDR_SIZE=" << hdr_size << endl;
    }    
  }

  if (hdr_size == 0 && !adjacent_header_file)
    throw Error (FailedCall, "dsp::DADAHeader::load_header", "HDR_SIZE undefined and no adjacent hdader file");
}

bool dsp::DADAHeader::empty () const
{
  return (header.size() == 0) || (header[0] == '\0');
}

bool dsp::DADAHeader::valid (bool verbose) const
{
  if (empty())
  {
    if (verbose)
      cerr << "dsp::DADAHeader::valid empty header" << endl;
    return false;
  }

  // verify that the buffer defines the HDR_VERSION
  float version;
  if (ascii_header_get (get_header(), "HDR_VERSION", "%f", &version) < 0)
  {
    if (verbose)
      cerr << "dsp::DADAHeader::valid HDR_VERSION not defined" << endl;
    return false;
  }

  // verify that the buffer defines the INSTRUMENT
  char instrument[64];
  if (ascii_header_get (get_header(), "INSTRUMENT", "%s", instrument) < 0)
  {
    if (verbose)
      cerr << "dsp::DADAHeader::valid INSTRUMENT not defined" << endl;
    return false;
  }

  return true;
}
