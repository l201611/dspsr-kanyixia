//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Kernel_Classes_SourceOpen_h
#define __dsp_Kernel_Classes_SourceOpen_h

#include "dsp/Source.h"

namespace dsp {

  //! Creates new Source objects
  class SourceOpen : public Reference::Able
  {

  public:

    //! Create new Source based on current command line options
    dsp::Source* open (int argc, char** argv);

    //! Print information about available inputs and unpackers
    static void list_backends();

    //! Command line values are header params, not file names
    bool command_line_header = false;

    //! Input files represent a single continuous observation
    bool force_contiguity = false;
  };

}

#endif // !defined(__dsp_Kernel_Classes_SourceOpen_h)
