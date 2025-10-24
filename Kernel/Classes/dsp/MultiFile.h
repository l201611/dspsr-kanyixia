//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/MultiFile.h

#ifndef __dsp_Kernel_Classes_MultiFile_h
#define __dsp_Kernel_Classes_MultiFile_h

#include "dsp/File.h"

namespace dsp {

  //! Base class of objects that load data from multiple files
  class MultiFile : public File {

  public:
  
    //! Constructor
    MultiFile (const char* name = "MultiFile");
    
    //! Destructor
    virtual ~MultiFile ();

    //! Returns true if filename is an ASCII file listing valid filenames
    bool is_valid (const char* filename) const;

    //! Return validated filenames loaded from metafile
    bool validate_filenames (std::vector<std::string>& filenames, const char* metafile) const;

    //! Open the files listed in the provided ASCII file
    virtual void open_file (const char* filename);

    //! Open a number of files and treat them as one logical observation composed of multiple parallel streams
    virtual void open (const std::vector<std::string>& new_filenames);

    //! Get the names of the loaded files
    const std::vector<std::string>& get_filenames() const { return filenames; }

    //! Inquire the number of files
    unsigned nfiles() { return files.size(); }

  protected:

    //! List of files
    std::vector< Reference::To<File> > files;

    //! Name of the currently opened file
    std::vector<std::string> filenames;

    //! Operation performed on each File instance after it is opened
    virtual void after_open (File*);

    //! Called at the end of open method, after all File instances are opened
    virtual void setup ();

  };

} // namespace dsp

#endif // !defined(__dsp_Kernel_Classes_MultiFile_h)
  
