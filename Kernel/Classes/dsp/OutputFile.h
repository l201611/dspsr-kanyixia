//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/OutputFile.h

#ifndef __OutputFile_h
#define __OutputFile_h

#include "dsp/Sink.h"
#include "dsp/BitSeries.h"

namespace dsp {

  //! Pure virtual base class of all objects that can unload BitSeries data
  /*!
    This class defines the common interface as well as some basic
    functionality relating to sources of BitSeries data.
  */

  class OutputFile : public Sink<BitSeries>
  {
  public:

    //! Constructor
    OutputFile (const char* operation_name);

    //! Destructor
    virtual ~OutputFile ();

    //! Close the current output file
    void close ();

    //! Set the number of fractional second decimal places to include in filename
    void set_fractional_second_decimal_places(unsigned);

  protected:

    friend class OutputFileShare;

    //! Unload data into the BitSeries specified with set_output
    virtual void calculation ();

    //! The file descriptor
    int fd = -1;
    
    //! The size of the header in bytes
    unsigned header_bytes = 0;

    //! The name of the currently open output file
    std::string output_filename;

    //! The pattern used to create an output filename
    std::string datestr_pattern = "%Y-%m-%d-%H:%M:%S";

    //! The number of fractional second decimal places to include in output filename
    unsigned fractional_second_decimal_places = 0;

    //! Return true if the input is valid
    virtual bool input_is_valid() const;

    //! Open the file specified by filename for writing
    virtual void open_file (const char* filename);

    //! Convenience wrapper
    void open_file (const std::string& name) { open_file (name.c_str()); }

    //! Write the file header to the open file
    virtual void write_header () = 0;

    //! Get the extension to be added to the end of new filenames
    virtual std::string get_extension () const = 0;

    //! Load nbyte bytes of sampled data from the device into buffer
    virtual int64_t unload_bytes (const void* buffer, uint64_t nbytes);
  };

}

#endif // !defined(__OutputFile_h)
