/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BitUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

using namespace std;

//! Null constructor
dsp::BitUnpacker::BitUnpacker (const char* _name)
  : HistUnpacker (_name)
{
  set_nstate (256);
}

dsp::BitUnpacker::~BitUnpacker ()
{
}

double dsp::BitUnpacker::get_optimal_variance ()
{
  if (!table)
    throw Error (InvalidState, "dsp::BitUnpacker::get_optimal_variance",
                 "BitTable not set");

  return table->get_optimal_variance();
}

void dsp::BitUnpacker::set_table (BitTable* _table)
{
  if (verbose)
    cerr << "dsp::BitUnpacker::set_table" << endl;

  table = _table;
}

const dsp::BitTable* dsp::BitUnpacker::get_table () const
{
  return table;
}

void dsp::BitUnpacker::unpack ()
{
  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  // Number of bytes to skip for each (set of) unpacked float(s) 
  // This assumes TFP-ordered input
  const unsigned nskip = npol * nchan * ndim;
  const unsigned fskip = ndim;

  auto nbit = input->get_nbit();
  if (nbit > 8)
    throw Error (InvalidState, "dsp::BitUnpacker::unpack",
                 "nbit=%d and current implementation works only for nbit <= 8", nbit);

  unsigned offset = 0;

  // by default, the histogram has a unique counter for each byte
  unsigned expected_hist_bins = 256;

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned idim=0; idim<ndim; idim++)
      {
        const unsigned char* from = input->get_rawptr() + offset;
        float* into = output->get_datptr (ichan, ipol) + idim;
        unsigned long* hist = get_histogram (offset, expected_hist_bins);

#ifdef _DEBUG
        cerr << "c=" << ichan << " p=" << ipol << " d=" << idim << endl;
#endif

        unpack_bits (ndat, from, nskip, into, fskip, hist);
        offset ++;
      }
    }
  }
}

