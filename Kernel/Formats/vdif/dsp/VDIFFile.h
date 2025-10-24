//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __VDIFFile_h
#define __VDIFFile_h

#include "dsp/BlockFile.h"

namespace dsp {

  //! Loads BitSeries data from a VDIF file
  /*! Loads data from a file containing raw VLBI Data Interchange Format 
   * (VDIF) packets. */
  class VDIFFile : public BlockFile
  {
  public:
	  
    //! Construct and open file	  
    VDIFFile (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~VDIFFile ();
	  
    //! Returns true if file starts with a valid VDIF packet header
    bool is_valid (const char* filename) const;

    //! Return true this this is contiguous with that
    bool contiguous (const File* that) const;

    //! For when the data file is not the current filename
    std::string get_data_filename () const { return datafile; }

  protected:

    friend class VDIFUnpacker;

    //! Open the file
    void open_file (const char* filename);

    //! Check that next packet follows the packet that was just read
    /*! Return the number of missing packets */
    uint64_t skip_extra ();

    //! Return the number of the next frame
    /*! \pre file pointer at start of next frame to be read */
    uint64_t get_next_frame_number ();

    void* stream;

    uint64_t reopen_seek;

#define VDIF_MAX_FILENAME_LENGTH 256

    char datafile[VDIF_MAX_FILENAME_LENGTH];

    int current_frame_number;
    int frames_per_second;
    
  };

}

#endif // !defined(__VDIFFile_h)
