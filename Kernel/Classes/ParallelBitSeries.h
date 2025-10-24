//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/ParallelBitSeries.h

#ifndef __dsp_Kernel_Classes_ParallelBitSeries_h
#define __dsp_Kernel_Classes_ParallelBitSeries_h

#include "dsp/BitSeries.h"

namespace dsp {

  //! Manages an array of BitSeries to be process in parallel
  class ParallelBitSeries : public Observation
  {

  public:
    
    //! Constructor
    ParallelBitSeries ();

    //! Create and manage N BitSeries objects
    void resize (unsigned N);

    //! Return the number of BitSeries objects
    unsigned size () const { return bitseries.size(); }

    //! Get the ith BitSeries instance
    BitSeries* at (unsigned i) { return bitseries.at(i); }
    const BitSeries* at (unsigned i) const { return bitseries.at(i); }

    void set_memory (Memory*);
    Memory* get_memory () { return memory; }

    //! Offset (owing to resolution) to the requested time sample
    unsigned get_request_offset () const;

    //! Number of time samples requested
    uint64_t get_request_ndat () const;

    //! Return the sample offset from the start of the data source
    int64_t get_input_sample (Input* input = 0) const;

    //! Copy the configuration of another Observation (not the data)
    void copy_configuration (const Observation* copy);

   protected:

    std::vector< Reference::To<BitSeries> > bitseries;

    //! The memory manager
    Reference::To<Memory> memory;
  };

} // namespace dsp

#endif // !defined(__dsp_Kernel_Classes_ParallelBitSeries_h)

