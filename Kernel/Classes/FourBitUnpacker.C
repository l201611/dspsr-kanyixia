/***************************************************************************
 *
 *   Copyright (C) 2008 - 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "environ.h"
#include "dsp/FourBitUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"
#include <assert.h>

using namespace std;

//! Null constructor
dsp::FourBitUnpacker::FourBitUnpacker (const char* _name)
  : BitUnpacker (_name)
{
  // with 4-bit nibbles, there are 16 output states ...
  set_nstate (16);

  // ... however, internally, count the frequency of 8-bit bytes
  set_nstate_internal (256);
}

void dsp::FourBitUnpacker::get_histogram (std::vector<unsigned long>& hist, unsigned idig) const
{
  assert( get_nstate() == 16 );
  assert( get_nstate_internal() == 256 );
  assert( idig < get_ndig() );

  hist.resize( get_nstate() );

  unsigned mask = 0x0f;

  const unsigned long* hist_internal = HistUnpacker::get_histogram (idig);

  /* Convert histogram of 8-bit bytes to histogram of 4-bit nibbles */
  for (unsigned i=0; i<get_nstate_internal(); i++)
  {
    unsigned s0 = i & mask;
    unsigned s1 = (i >> 4) & mask;

    hist[s0] += hist_internal[i];
    hist[s1] += hist_internal[i];
  }
}


void dsp::FourBitUnpacker::unpack_bits (
  uint64_t ndat,
  const unsigned char* from,
  const unsigned nbyte_stride,
  float* into,
  const unsigned nfloat_stride,
  unsigned long* hist
)
{
  const uint64_t ndat2 = ndat/2;
  const float* lookup = table->get_values ();

  if (ndat % 2)
    throw Error (InvalidParam, "dsp::FourBitUnpacker::unpack",
                 "invalid ndat=" UI64, ndat);

  for (uint64_t idat = 0; idat < ndat2; idat++)
  {
    /* Note that the histogram counts the frequency of 8-bit bytes.
       These are converted to histograms of 4-bit nibbles in
       FourBitUnpacker::get_histogram.  This works only under the
       assumption that each byte encodes a single process. */

    hist[ *from ] ++;

    /* Each byte encodes two samples by addressing two consecutive floats
       in a lookup table of 256 pairs of pre-computed floating point values. */
    unsigned offset = unsigned(*from) * 2;
    from += nbyte_stride;

    *into = lookup[ offset ];
    into += nfloat_stride;
    *into = lookup[ offset + 1 ];
    into += nfloat_stride;
  }
}
