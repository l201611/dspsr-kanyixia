//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Kernel_Classes_SingleFile_h
#define __dsp_Kernel_Classes_SingleFile_h

#include "dsp/ParallelInput.h"

namespace dsp {

  //! Test the ParallelInput concept using a single File
  class SingleFile : public ParallelInput
  {
  public:
    SingleFile ();
    ~SingleFile ();

    //! Return true if descriptor describes a file that can be opened
    bool matches (const std::string& descriptor) const override;

    //! Open the file
    void open (const std::string& descriptor) override;

    //! Load BitSeries data
    void load (ParallelBitSeries*) override;

    //! Return the number of time samples to load on each load_block
    uint64_t get_block_size () const override;
    //! Set the number of time samples to load on each load_block
    void set_block_size (uint64_t _size) override;

    //! Return the number of time samples by which consecutive blocks overlap
    uint64_t get_overlap () const override;
    //! Set the number of time samples by which consecutive blocks overlap
    void set_overlap (uint64_t _overlap) override;

    //! Get the time sample resolution of the data source
    unsigned get_resolution () const override;
  };

}

#endif // !defined(__dsp_Kernel_Classes_SingleFile_h)
