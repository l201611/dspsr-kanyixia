//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/Expect.h

#ifndef __dsp_Kernel_Classes_Expect_h
#define __dsp_Kernel_Classes_Expect_h

#include "FilePtr.h"

//! Checks for expected string in a data file
class Expect
{
  protected:
    //! Pointer to the open file
    FilePtr file;

  public:

    //! Open filename for reading
    Expect(const std::string& filename);

    //! Return true if the next string read from file matches text
    bool expect(const std::string& text);

    //! Get the pointer to the open file
    FILE* fptr();
};

#endif
