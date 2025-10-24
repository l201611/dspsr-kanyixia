//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 - 2023 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/DualFile.h

#ifndef __dsp_Kernel_Classes_DualFile_h
#define __dsp_Kernel_Classes_DualFile_h

#include "dsp/ParallelBitSeries.h"
#include "dsp/ParallelInput.h"

namespace dsp {

  /**
   * @brief Load a pair of input BitSeries using ParallelBitSeries from a pair of parallel Inputs.
   * This class extends the ParallelInput based class is configured by a descriptor which 
   * defines a subsequent pair of singular Inputs. The singular Inputs support classes derived from
   * dsp::File.
   * 
   */
  class DualFile : public ParallelInput {

  public:
  
    //! Constructor
    DualFile ();
    
    //! Destructor
    ~DualFile ();
    
    //! Return true if descriptor describes a parallel file that can be opened
    bool matches (const std::string& descriptor) const override;

    //! Open the file
    void open (const std::string& descriptor) override;

    //! Set the pair of inputs
    virtual void set_inputs (Input* first, Input* second);

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

  protected:

  private:

  };

}

#endif // !defined(__dsp_Kernel_Classes_DualFile_h)
  
