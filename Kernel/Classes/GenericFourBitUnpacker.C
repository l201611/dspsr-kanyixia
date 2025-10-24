/***************************************************************************
 *
 *   Copyright (C) 2017-2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericFourBitUnpacker.h"
#include "dsp/BitTable.h"

#include <iostream>
using namespace std;

dsp::GenericFourBitUnpacker::GenericFourBitUnpacker ()
  : FourBitUnpacker ("GenericFourBitUnpacker")
{
#define ASSUME_TWOS_COMPLEMENT 1
#if ASSUME_TWOS_COMPLEMENT
  BitTable* table = new BitTable (4, BitTable::TwosComplement);
#else
  BitTable* table = new BitTable (4, BitTable::OffsetBinary);
#endif
  table->set_order( BitTable::LeastToMost );
  set_table( table );
}

bool dsp::GenericFourBitUnpacker::matches (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::GenericUnpacker::matches"
      " machine=" << observation->get_machine() <<
      " nbit=" << observation->get_nbit() << endl;

  return observation->get_nbit() == 4;
}

void dsp::GenericFourBitUnpacker::unpack ()
{
  uint64_t ndat  = input->get_ndat();
  unsigned nchan = input->get_nchan();
  unsigned npol  = input->get_npol();
  unsigned ndim  = input->get_ndim();

  unsigned nfloat_per_time_slice = nchan * npol * ndim;
  unsigned nfloat_per_byte = 2;

  if (nfloat_per_time_slice % nfloat_per_byte != 0)
  {
    throw Error (InvalidState, "dsp::GenericFourBitUnpacker::unpack",
                "number of floats to unpack per time slice = %u is not divisible by %u",
                nfloat_per_time_slice, nfloat_per_byte);
  }

  // Number of bytes to skip for each pair of floats (assumes TFP-ordered input bits)
  unsigned nbyte_stride = nfloat_per_time_slice / nfloat_per_byte;

  // Happens only when nchan = npol = ndim = 1
  if (nbyte_stride == 0)
  {
    nbyte_stride = 1;
  }

  uint64_t nfloat_stride = 1;
  unsigned pol_step = 1;
  unsigned chan_step = 1;

  if (ndim > 1)
  {
    DEBUG("GenericFourBitUnpacker::unpack Re,Im per byte");
  }
  else if (npol > 1)
  {
    DEBUG("GenericFourBitUnpacker::unpack two pol per byte");
    pol_step = nfloat_per_byte;  // unpack two polarizations from each byte
    nfloat_stride = output->get_stride();
  }
  else if (nchan > 1)
  {
    DEBUG("GenericFourBitUnpacker::unpack two chan per byte");
    chan_step = nfloat_per_byte; // unpack two frequencies from each byte
    nfloat_stride = output->get_stride();
  }
  else
  {
    DEBUG("GenericFourBitUnpacker::unpack two dat per byte");
    if (ndat % nfloat_per_byte != 0)
    {
      throw Error (InvalidState, "dsp::GenericFourBitUnpacker::unpack",
                  "number of time samples = %lu is not divisible by %u",
                  ndat, nfloat_per_byte);
    }
    ndat /= nfloat_per_byte; // unpack two time samples from each byte
  }

  unsigned offset = 0;

  DEBUG("GenericFourBitUnpacker::unpack stride input=" << nbyte_stride << " output=" << nfloat_stride);

  // by default, the histogram has a unique counter for each byte
  unsigned expected_hist_bins = 256;

  for (unsigned ichan=0; ichan < nchan; ichan += chan_step)
  {
    for (unsigned ipol=0; ipol < npol; ipol += pol_step)
    {
      const unsigned char* from = input->get_rawptr() + offset; 
      float* into = output->get_datptr (ichan, ipol);
      unsigned long* hist = HistUnpacker::get_histogram (offset, expected_hist_bins);

      unpack_bits (ndat, from, nbyte_stride, into, nfloat_stride, hist);
      offset ++;
    }
  }
}

void dsp::GenericFourBitUnpacker::unpack_bits
( uint64_t ndat,
  const unsigned char* from,
  const unsigned nbyte_stride,
  float* into,
  const unsigned nfloat_stride,
  unsigned long* hist
)
{
  const float* lookup = table->get_values ();

  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    /* Note that the histogram counts the frequency of 8-bit bytes.
    These are converted to histograms of 4-bit nibbles in
    FourBitUnpacker::get_histogram.

    TO FIX: This trick incorrectly assumes that each byte encodes a single process,
    which is true only when nchan = npol = ndim = 1.  If ndim=2, then the two histograms
    for the real and imaginary parts of each process will be combined into one.
    */

    hist[ *from ] ++;

    /* Each byte encodes two samples by addressing two consecutive floats
    in a lookup table of 256 pairs of pre-computed floating point values. */

    auto offset = unsigned(*from) * 2;
    from += nbyte_stride;

    *into = lookup[ offset ];
    into += nfloat_stride;
    *into = lookup[ offset + 1 ];
    into += nfloat_stride;
  }
}
