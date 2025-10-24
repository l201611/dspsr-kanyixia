//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Jonathon Kocz
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/Multiplex.h

#ifndef __Multiplex_h
#define __Multiplex_h

#include "dsp/MultiFile.h"

namespace dsp {

  //! Loads BitSeries data from multiple files
  class Multiplex : public MultiFile {

  public:
  
    //! Constructor
    Multiplex ();
    
    //! Destructor
    virtual ~Multiplex ();

  protected:

    //! Load bytes from file
    virtual int64_t load_bytes (unsigned char* buffer, uint64_t bytes);
    
    //! Adjust the file pointer
    virtual int64_t seek_bytes (uint64_t bytes);

    //! Currently open File instance
    Reference::To<File> loader;

    //! Index of the current File in use
    unsigned current_index;
  };

}

#endif // !defined(__Multiplex_h)
  
