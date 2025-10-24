//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 - 2023 by Aidan Hotan and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/IOManager.h

#ifndef __dsp_Kernel_Classes_IOManager_h
#define __dsp_Kernel_Classes_IOManager_h

#include "dsp/UnpackerSource.h"
#include "dsp/Input.h"
#include "dsp/Unpacker.h"

namespace dsp {

  //! Produces TimeSeries data by integrating an Input with an Unpacker
  class IOManager : public UnpackerSource<Unpacker,Input>
  {
  public:

    //! Constructor
    IOManager () : UnpackerSource<Unpacker,Input>("IOManager") {}

    //! Return a default constructed clone of self
    IOManager* clone() const override;

    //! Prepare the appropriate Input and Unpacker
    virtual void open (const std::string& id);
  };

} // namespace dsp

#endif // !defined(__dsp_Kernel_Classes_IOManager_h)
