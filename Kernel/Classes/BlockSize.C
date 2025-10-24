/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "environ.h"
#include "dsp/BlockSize.h"
#include "ascii_header.h"
#include "Error.h"

using namespace std;

void dsp::BlockSize::load (const char* header)
{
  if (header == NULL)
    throw Error (InvalidState, "dsp::load_dada_header", "header == NULL");

  // //////////////////////////////////////////////////////////////////////
  //
  // BLOCK_DATA_BYTES
  //
  if (ascii_header_get (header, "BLOCK_DATA_BYTES", UI64, &block_data_bytes) < 0)
  {
    throw Error (InvalidParam, "dsp::BlockSize::load", "header does not define BLOCK_DATA_BYTES");
  }

  // //////////////////////////////////////////////////////////////////////
  //
  // BLOCK_HEADER_BYTES
  //
  bool has_header = ascii_header_get (header, "BLOCK_HEADER_BYTES", UI64, &block_header_bytes) == 1;
  if (!has_header)
    block_header_bytes = 0;

  // //////////////////////////////////////////////////////////////////////
  //
  // BLOCK_TAILER_BYTES
  //
  bool has_tailer = ascii_header_get (header, "BLOCK_TAILER_BYTES", UI64, &block_tailer_bytes) == 1;
  if (!has_tailer)
    block_tailer_bytes = 0;

  if (!has_header && !has_tailer)
  {
    throw Error (InvalidParam, "dsp::BlockSize::load", "header defines neither BLOCK_HEADER_BYTES nor BLOCK_TAILER_BYTES");
  }
}


dsp::BlockSize* dsp::BlockSize::clone () const
{
  if (Observation::verbose)
    cerr << "dsp::BlockSize::clone" << endl;

  return new BlockSize(*this);
}

uint64_t dsp::BlockSize::get_nbytes (uint64_t nsamples) const
{
  if (Observation::verbose)
    cerr << "dsp::BlockSize::get_nbytes" << endl;

  uint64_t data_bytes = Observation::NbyteNsamplePolicy::get_nbytes(nsamples);
  if (block_data_bytes == 0)
    return data_bytes;

  uint64_t nblock = data_bytes / block_data_bytes;

  if (data_bytes % block_data_bytes)
    nblock ++;

  if (Observation::verbose)
    cerr << "dsp::BlockSize::get_nbytes return " << nblock * get_block_bytes() << " data_bytes=" << data_bytes << endl;

  return nblock * get_block_bytes();
}

uint64_t dsp::BlockSize::get_nsamples (uint64_t nbytes) const
{
  if (Observation::verbose)
    cerr << "dsp::BlockSize::get_nsamples" << endl;

  if (block_data_bytes == 0)
    return Observation::NbyteNsamplePolicy::get_nsamples(nbytes);

  uint64_t block_bytes = get_block_bytes();
  uint64_t nblock = nbytes / block_bytes;
  return Observation::NbyteNsamplePolicy::get_nsamples(nblock * block_data_bytes);
}
