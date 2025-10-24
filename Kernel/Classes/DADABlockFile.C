/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/DADABlockFile.h"
#include "dsp/ASCIIObservation.h"
#include "dsp/BlockSize.h"

#include "ascii_header.h"
#include "FilePtr.h"
#include "Error.h"
#include "strutil.h"

#include <fstream>
#include <fcntl.h>

using namespace std;

dsp::DADABlockFile::DADABlockFile (const char* filename) : BlockFile ("DADABlock")
{
  if (filename)
    open (filename);
}

bool dsp::DADABlockFile::is_valid (const char* filename) const try
{
  DADAHeader tmp_hdr;
  tmp_hdr.load_header (filename);

  // //////////////////////////////////////////////////////////////////////
  //
  // BLOCK_KEEP_ONLY_DATA
  //
  // In some files with BLOCK_*_BYTES defined in the header, the block 
  // headers (and tailers) should be kept.  The DADABlockFile class should
  // be used when only the data should be kept.
  //
  int keep_only_data = 0;
  bool has_value = ascii_header_get (tmp_hdr.get_header(), "BLOCK_KEEP_ONLY_DATA", "%d", &keep_only_data) == 1;
  if (! (has_value && keep_only_data))
  {
    if (verbose)
      cerr << "dsp::DADABlockFile::is_valid BlockSize BLOCK_KEEP_ONLY_DATA is not set to 1" << endl;
    return false;
  }

  if (!tmp_hdr.valid(verbose))
  {
    if (verbose)
      cerr << "dsp::DADABlockFile::is_valid DADAHeader::is_valid returns false" << endl;
    return false;
  }

  BlockSize block_size;
  block_size.load(tmp_hdr.get_header());

  if (verbose)
    cerr << "dsp::DADABlockFile::is_valid BlockSize successfully parsed" << endl;

  return true;
}
catch (Error& error)
{
  if (verbose)
    cerr << "dsp::DADABlockFile::is_valid " << error.get_message() << endl;
  return false;
}

void dsp::DADABlockFile::open_file (const char* filename)
{
  if (verbose)
    cerr << "DADABlockFile::open filename=" << filename << endl;

  dada_header.load_header (filename);

  if (dada_header.empty())
    throw Error (FailedCall, "dsp::DADABlockFile::open_file", "get_header(%s) failed", filename);

  auto obs = new ASCIIObservation(dada_header.get_header());
  info = obs;

  header_bytes = dada_header.get_header_size();
  resolution = obs->get_resolution();

  auto block_size = dynamic_cast<const BlockSize*>(info->get_nbyte_nsample_policy());
  if (!block_size)
  {
    throw Error (InvalidState, "dsp::DADABlockFile::open_file", "no BlockSize");
  }

  //! Total number of bytes in each block (header + data + tailer)
  block_bytes = block_size->get_block_bytes();
  block_header_bytes = block_size->get_block_header_bytes();
  block_tailer_bytes = block_size->get_block_tailer_bytes();

  // the Observation base class will set the policy back to the default
  info->set_nbyte_nsample_policy (nullptr);

  open_fd (filename);

  if (verbose)
    cerr << "DADABlockFile::open exit" << endl;
}
