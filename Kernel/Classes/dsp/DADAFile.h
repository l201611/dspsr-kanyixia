//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/DADAFile.h

#ifndef __DADAFile_h
#define __DADAFile_h

#include "dsp/File.h"
#include "dsp/DADAHeader.h"

namespace dsp {

  //! Loads BitSeries data from a DADA data file
  class DADAFile : public File 
  {

  public:
   
    //! Construct and open file
    DADAFile (const char* filename=0);

    //! Returns true if filename appears to name a valid DADA file
    bool is_valid (const char* filename) const override;

    //! Return the immutable DADA ASCII header loaded from filename
    const char* get_header () const { return dada_header.get_header(); }

  protected:

    //! Open the file
    void open_file (const char* filename) override;

    //! ASCII header loaded from file
    DADAHeader dada_header;
  };

}

#endif // !defined(__DADAFile_h)
  
