//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/BitUnpacker.h

#ifndef __BitUnpacker_h
#define __BitUnpacker_h

#include "dsp/HistUnpacker.h"

namespace dsp {

  class BitTable;

  //! Converts N-bit digitised samples to floating point using a BitTable
  /*! This unpacker assumes that nbit <= 8; and each byte contains N=8/nbit 
    consecutive time samples from a single real-valued process.

    For example, if nbit=2 and ndim=2, then the real parts of 4 consecutive 
    time samples are packed into one byte, and the imaginary parts of the 
    same 4 time samples are packed into the following byte. 

    The bytes for a given real-valued process are interleaved with the
    bytes of other processes in time, frequency, polarization, dimension
    (TFPD) order; e.g. if ndim=2, npol=2, and nbit=8, then
    the first few bytes contain

    t0f0p0d0, t0f0p0d1, t0f0p1d0, t0f0p1d1, t0f1p0d0, ... t1f0p0d0, ...

    If these assumptions are not valid for a given file format, then 
    the unpack method should be over-ridden in a derived class that handles
    that file format. 
    */
  class BitUnpacker: public HistUnpacker
  {

  public:

    //! Null constructor
    BitUnpacker (const char* name = "BitUnpacker");

    //! Virtual destructor
    virtual ~BitUnpacker ();

    //! Get the optimal value of the output time series variance
    double get_optimal_variance ();

    //! Set the digitisation convention
    void set_table (BitTable* table);

    //! Get the digitisation convention
    const BitTable* get_table () const;

  protected:

    //! Unpack all channels, polarizations, and dimensions as separate processes
    /*! Calls unpack_bits for each frequency channel, polarization, and dimension. */
    virtual void unpack ();

    //! Unpack a single real-valued process (e.g. single digitizer output)
    /*!
    Input bytes are interleaved in time, frequency, polarization, dimension (TFPD) order.
    Each byte contains N=8/nbit consecutive time samples from a single real-valued process.

    @param ndat total number of real-valued time samples
    @param from base address of TFPD-interleaved digitized samples
    @param from_byte_stride byte offset between consecutive digitized sample(s)
    @param into base address of unpacked time-contiguous floating point data
    @param into_float_stride offset to next consecutive real-valued sample
    @param hist base address of histogram of digitized states
    */
    virtual void unpack_bits
    (
      uint64_t ndat,
      const unsigned char* from,
      const unsigned from_byte_stride,
      float* into,
      const unsigned into_float_stride,
      unsigned long* hist
    ) = 0;

    //! The bit table generator
    Reference::To<BitTable> table;
  };

}

#endif

