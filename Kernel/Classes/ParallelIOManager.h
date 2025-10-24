//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/ParallelIOManager.h

#ifndef __dsp_Kernel_Classes_ParallelIOManager_h
#define __dsp_Kernel_Classes_ParallelIOManager_h

#include "dsp/UnpackerSource.h"
#include "dsp/ParallelInput.h"
#include "dsp/ParallelUnpacker.h"

namespace dsp {

  //! Produces TimeSeries data by integrating a ParallelInput with a ParallelUnpacker
  class ParallelIOManager : public UnpackerSource<ParallelUnpacker,ParallelInput>
  {

  public:

    //! Constructor
    ParallelIOManager () : UnpackerSource<ParallelUnpacker,ParallelInput>("ParallelIOManager") {}

    //! Return a default constructed clone of self
    ParallelIOManager* clone() const override;

    //! Prepare the appropriate ParallelInput and ParallelUnpacker
    virtual void open (const std::string& id);

  };

} // namespace dsp

#endif // !defined(__dsp_Kernel_Classes_ParallelIOManager_h)
