/***************************************************************************
 *
 *   Copyright (C) 2023 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SixteenBitUnpacker.h"

#include "Error.h"

using namespace std;

//! Null constructor
dsp::SixteenBitUnpacker::SixteenBitUnpacker (BitTable::Type _encoding, const char* _name)
  : HistUnpacker (_name), encoding(_encoding)
{
  if (verbose)
    cerr << "dsp::SixteenBitUnpacker ctor" << endl;
  set_nstate (65536);
}

void dsp::SixteenBitUnpacker::unpack ()
{
  const uint64_t ndat  = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();

  const unsigned nskip = npol * nchan * ndim;
  const unsigned fskip = ndim;

  unsigned offset = 0;

  // by default, the histogram has a unique counter for each word
  unsigned expected_hist_bins = 65536;

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned idim=0; idim<ndim; idim++)
      {
	      const int16_t * from = reinterpret_cast<const int16_t *>(input->get_rawptr()) + offset;
        float * into = output->get_datptr (ichan, ipol) + idim;
	      unsigned long* hist = get_histogram (offset,expected_hist_bins);
        unpack_bits (ndat, from, nskip, into, fskip, hist);
        offset ++;
      }
    }
  }
}

void dsp::SixteenBitUnpacker::unpack_bits (uint64_t ndat,
	const int16_t * from,
	const unsigned nskip,
	float* into,
  const unsigned fskip,
	unsigned long* hist)
{
  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    int16_t fromval = *from;
    if (encoding == BitTable::Type::OffsetBinary)
      fromval = fromval ^ 0x8000;

    // add 32k because fromval is signed
    hist[ fromval+32768 ] ++;
    *into = float(fromval) + 0.5;

#ifdef _DEBUG
    if (idat < 32)
      cerr << idat << " *from=" << int(*from) << " fromval=" << fromval << " into=" << *into << endl;
#endif

    from += nskip;
    into += fskip;
  }
}
