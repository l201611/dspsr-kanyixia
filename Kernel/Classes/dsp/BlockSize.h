//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/BlockSize.h

#ifndef __dsp_BlockSize_h
#define __dsp_BlockSize_h

#include "dsp/Observation.h"

namespace dsp {

  //! Sizes of header, data, and tailer of data that are divided in blocks / frames / heaps
  class BlockSize : public dsp::Observation::NbyteNsamplePolicy
  {
  public:

    //! Return a new instance of self
    BlockSize* clone () const override;

    //! Return the size in bytes of nsamples time samples
    uint64_t get_nbytes (uint64_t nsamples) const override;

    //! Return the number of samples in nbytes bytes
    uint64_t get_nsamples (uint64_t nbytes) const override;

    //! Get the number of data bytes per block
    uint64_t get_block_data_bytes() const { return block_data_bytes; }

    //! Set the number of data bytes per block
    void set_block_data_bytes(uint64_t bytes) { block_data_bytes = bytes; }

    //! Get the number of bytes in header of each block
    uint64_t get_block_header_bytes() const { return block_header_bytes; }

    //! Set the number of bytes in header of each block
    void set_block_header_bytes(uint64_t bytes) { block_header_bytes = bytes; }

    //! Get the number of bytes in tailer of each block
    uint64_t get_block_tailer_bytes() const { return block_tailer_bytes; }

    //! Set the number of bytes in tailer of each block
    void set_block_tailer_bytes(uint64_t bytes) { block_tailer_bytes = bytes; }

    //! Get the total number of bytes in each block (header + data + tailer)
    uint64_t get_block_bytes() const { return block_header_bytes + block_data_bytes + block_tailer_bytes; }

    //! Load BlockSize attributes from a DADA ASCII header
    void load (const char* header);

  protected:

    //! Number of bytes of data in each block
    uint64_t block_data_bytes {0};

    //! Number of bytes in header of each block
    uint64_t block_header_bytes {0};

    //! Number of bytes in tailer of each block
    uint64_t block_tailer_bytes {0};
  };
}

#endif // !defined(__BlockSize_h)
  

