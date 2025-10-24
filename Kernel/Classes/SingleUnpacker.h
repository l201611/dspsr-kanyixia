//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Kernel_Classes_SingleUnpacker_h
#define __dsp_Kernel_Classes_SingleUnpacker_h

#include "dsp/ParallelUnpacker.h"

namespace dsp {

  //! Test the ParallelUnpacker concept using a single Unpacker
  class SingleUnpacker : public ParallelUnpacker
  {
  public:

    //! Default constructor
    SingleUnpacker ();

    //! Destructor
    ~SingleUnpacker ();

    //! Return true if descriptor describes a file that can be opened
    bool matches (const Observation* observation) const override;

    //! Specialize the unpackers for the Observation
    void match (const Observation* observation) override;

    //! Match the unpackers to the resolution
    void match_resolution (ParallelInput*) override;

    //! Return the smallest number of time samples that can be unpacked
    unsigned get_resolution () const override;

  protected:

    //! The unpacking routine
    virtual void unpack () override;
  };

}

#endif // !defined(__dsp_Kernel_Classes_SingleUnpacker_h)
