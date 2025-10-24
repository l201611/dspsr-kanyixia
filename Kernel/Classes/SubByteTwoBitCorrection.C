/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SubByteTwoBitCorrection.h"
#include "dsp/excision_unpack.h"
#include "dsp/StepIterator.h"

#include <iostream>
#include <algorithm>

using namespace std;

// #define _DEBUG 1

dsp::SubByteTwoBitCorrection::SubByteTwoBitCorrection (const char* name)
  : TwoBitCorrection (name)
{
}

dsp::SubByteTwoBitCorrection::~SubByteTwoBitCorrection ()
{
}

/*! By default, both polarizations are output in one byte */
unsigned dsp::SubByteTwoBitCorrection::get_ndig_per_byte () const
{
  unsigned max_ndig = 4;
  return std::min (get_ndig(), max_ndig);
}

/*! By default, the data is not interleaved byte by byte */
unsigned dsp::SubByteTwoBitCorrection::get_input_offset (unsigned idig) const
{
  return idig / get_ndig_per_byte ();
}

/*! By default, the data is not interleaved byte by byte */
unsigned dsp::SubByteTwoBitCorrection::get_input_incr () const 
{
  return get_ndig() / get_ndig_per_byte ();
}

/*! By default, MSB y1 x1 y0 x0 LSB */
unsigned
dsp::SubByteTwoBitCorrection::get_shift (unsigned idig, unsigned isamp) const
{
#if _DEBUG
  cerr << "dsp::SubByteTwoBitCorrection::get_shift idig=" << idig;
#endif

  unsigned shift = (idig % get_ndig_per_byte () + isamp * 2) * 2;

#if _DEBUG
  cerr << " isamp=" << isamp << " shift=" << shift << endl;
#endif

  assert (shift < 7);
  return shift;
}

/* By default, there may be one time sample from each of two or four
   digitizer outputs in each byte.  For an example of code with two
   samples from each of two digitizers, with bits ordered in a
   different way, please see mark5/Mark5TwoBitCorrection.C. */
void dsp::SubByteTwoBitCorrection::dig_unpack (const unsigned char* input_data,
					       float* output_data,
					       uint64_t nfloat,
					       unsigned long* hist,
					       uint16_t* weights,
					       unsigned nweights)
{
  StepIterator<const unsigned char> iterator (input_data);
  iterator.set_increment ( get_input_incr() );

  const unsigned ndig_per_byte = get_ndig_per_byte();

#if ! _DEBUG
  if (verbose)
#endif
    cerr << "dsp::SubByteTwoBitCorrection::dig_unpack"
            " input_data=" << (void*) input_data <<
            " input_incr=" << get_input_incr() << 
            " ndig_per_byte=" << ndig_per_byte << endl <<
            " nfloat=" << nfloat << " nweight=" << nweights << endl;

  if (ndig_per_byte == 2)
  {
    // unpack 2 samples per byte

    unpack2.mask.shift[0] = get_shift (current_digitizer, 0);
    unpack2.mask.shift[1] = get_shift (current_digitizer, 1);

    ExcisionUnpacker::excision_unpack (unpack2, iterator,
				       output_data, nfloat,
				       hist, weights, nweights);
  }
  else if (ndig_per_byte == 4)
  {
    // unpack 1 sample per byte

    unpack1.mask.shift[0] = get_shift (current_digitizer, 0);

    ExcisionUnpacker::excision_unpack (unpack1, iterator,
				       output_data, nfloat,
				       hist, weights, nweights);
  }
  else
    throw Error (InvalidState, "dsp::SubByteTwoBitCorrection::dig_unpack",
		 "invalid number of digitizers per byte: %u", ndig_per_byte);
}

dsp::TwoBitLookup* dsp::SubByteTwoBitCorrection::get_unpacker ()
{
  switch (get_ndig_per_byte())
  {
  case 2:
    return &unpack2;
  case 4:
    return &unpack1;
  default:
    throw Error (InvalidState, "dsp::SubByteTwoBitCorrection::get_unpacker",
		 "invalid number of digitizers per byte: %u",
		 get_ndig_per_byte());
  }
}

static unsigned mask_lsb = 1;

unsigned dsp::SubByteTwoBitCorrection::get_output_ipol (unsigned idig) const
{
#if _DEBUG
  cerr << "dsp::SubByteTwoBitCorrection::get_output_ipol idig=" << idig;
#endif

  if (input->get_npol() == 1)
    return 0;
  
  if (input->get_ndim() > 1)
    idig = idig >> 1;

#if _DEBUG
  cerr << " ipol=" << (idig & mask_lsb) << endl;
#endif
 
  return idig & mask_lsb;
}

unsigned dsp::SubByteTwoBitCorrection::get_output_ichan (unsigned idig) const
{
#if _DEBUG
  cerr << "dsp::SubByteTwoBitCorrection::get_output_ichan idig=" << idig;
#endif

  if (input->get_nchan() == 1)
    return 0;

  if (input->get_ndim() > 1)
    idig = idig >> 1;

  if (input->get_npol() > 1)
    idig = idig >> 1;

#if _DEBUG
  cerr << " ichan=" << idig << endl;
#endif

  return idig;
}


/*! The quadrature components must be offset by one */
unsigned dsp::SubByteTwoBitCorrection::get_output_offset (unsigned idig) const
{
  if (input->get_ndim() == 1)
    return 0;
  else
    return idig & mask_lsb;
}

/*! The in-phase and quadrature components must be interleaved */
unsigned dsp::SubByteTwoBitCorrection::get_output_incr () const
{
  return input->get_ndim();
}

