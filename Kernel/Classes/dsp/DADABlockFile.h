//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/DADABlockFile.h

#ifndef __DADABlockFile_h
#define __DADABlockFile_h

#include "dsp/BlockFile.h"
#include "dsp/DADAHeader.h"

namespace dsp {

  //! Loads BitSeries data from a DADA data file in which data are organized in blocks
  /*! Only the data in each block are kept.  Header and tailer bytes are discarded. */
  class DADABlockFile : public BlockFile
  {

  public:

    //! Construct and open file
    DADABlockFile (const char* filename=0);

    //! Returns true if filename appears to name a valid DADA file
    /*! BLOCK_KEEP_ONLY_DATA must be set to 1 in the header. */
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

#endif // !defined(__DADABlockFile_h)
