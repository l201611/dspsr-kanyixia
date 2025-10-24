//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Kernel_Classes_SourceFactory_h
#define __dsp_Kernel_Classes_SourceFactory_h

#include "Source.h"

namespace dsp {

  //! Creates new Source objects
  class SourceFactory : public Reference::Able
  {
  public:

    //! Construct a new child of Source based on the descriptor
    dsp::Source* create (const std::string& descriptor);

  };

}

#endif // !defined(__dsp_Kernel_Classes_SourceFactory_h)
