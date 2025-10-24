//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 - 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/SerialFiles.h

#ifndef __dsp_Kernel_Classes_SerialFiles_h
#define __dsp_Kernel_Classes_SerialFiles_h

#include "dsp/MultiFile.h"

namespace dsp {

  //! Loads serial BitSeries data from multiple files
  class SerialFiles : public MultiFile {

  public:
  
    //! Constructor
    SerialFiles (const char* name = "SerialFiles");
    
    //! Destructor
    virtual ~SerialFiles ();

    //! The origin is the current loader
    const Input* get_origin () const { return get_loader(); }

    //! Treat the files as contiguous
    void force_contiguity ();

    //! Makes sure only these filenames are open
    virtual void have_open (const std::vector<std::string>& filenames);

    //! Retrieve a pointer to the loader File instance
    File* get_loader ();
    const File* get_loader () const;

    //! Access to current file objects
    std::vector< Reference::To<File> >& get_files () {return files;}

    //! Return true if the loader File instance is set
    bool has_loader ();

    //! Erase the entire list of loadable files
    //! Resets the file pointers
    virtual void erase_files();

    //! Erase just some of the list of loadable files
    //! Resets the file pointers regardless
    virtual void erase_files (const std::vector<std::string>& erase_filenames);

    //! Find out which file is currently open;
    std::string get_current_filename() const { return current_filename; }

    //! Find out the index of current file is
    unsigned get_index() const { return current_index; }

    //! Inquire the next sample to load for the current file
    uint64_t get_next_sample();

    //! Add any relevant extensions (calls loader's add_extensions())
    void add_extensions (Extensions *ext);

  protected:

    //! Load bytes from file
    virtual int64_t load_bytes (unsigned char* buffer, uint64_t bytes);
    
    //! Adjust the file pointer
    virtual int64_t seek_bytes (uint64_t bytes);

    //! Currently open File instance
    Reference::To<File> loader;

    //! Name of the currently opened file
    std::string current_filename;

    //! initialize variables
    void init();

    //! Ensure that files are contiguous
    void ensure_contiguity ();

  private:

    //! Test for contiguity
    bool test_contiguity;

    //! Index of the current File in use
    unsigned current_index;

    //! Close each File after it is opened
    void after_open (File*);

    //! Setup loader and ndat etc after a change to the list of files
    void setup ();

    //! Set the loader to the specified File
    void set_loader (unsigned index);

  };

}

#endif // !defined(__dsp_Kernel_Classes_SerialFiles_h)
  
