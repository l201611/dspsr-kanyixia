//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 - 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __DADABuffer_h
#define __DADABuffer_h

#include "dsp/File.h"
#include "dada_hdu.h"

namespace dsp {

  //! Loads BitSeries data from a DADA ring buffer shared memory
  /*! This class pretends to be a file so that it can slip into the
    File::registry */
  class DADABuffer : public File
  {

  public:

    //! Constructor
    DADABuffer ();

    //! Destructor
    ~DADABuffer ();

    //! Returns true if the ASCII contents of file describe a valid DADA ring buffer
    /*! The first two lines of the file must match

    \code
    DADA INFO:
    key XXXX
    \endcode

    where XXXX is a four-character hexadecimal string.
    */
    bool is_valid (const char* filename) const override;

    //! Read the DADA key information from the specified filename
    void open_file (const char* filename) override;

    //! Open the DADA ring buffer via the key
    virtual void open_key (const std::string& key);

    //! Load the next header block and parse it
    void load_header ();

    //! Re-open using the current key
    /*! Calls load_header and rewind */
    void reopen () override;

    //! Close the DADA connection
    void close () override;

    //! Seek to the specified time sample
    void seek (int64_t offset, int whence = 0) override;

    //! Ensure that block_size is an integer multiple of resolution
    void set_block_size (uint64_t _size) override;

    //! End-of-data is defined by primary read client (passive viewer)
    bool eod() const override;

    //! Get the information about the data source
    virtual void set_info (Observation* obs) { info = obs; }

    //! Reset DADAbuffer
    void rewind () override;

  protected:
    //! Disconnect from the DADA ring buffer
    void disconnect ();

    //! Load bytes from shared memory
    int64_t load_bytes (unsigned char* buffer, uint64_t bytes) override;

    //! Set the offset in shared memory
    int64_t seek_bytes (uint64_t bytes) override;

    //! Over-ride File::set_total_samples
    void set_total_samples () override;

    //! Shared memory interface
    dada_hdu_t* hdu = nullptr;

    //! Passive viewing mode
    bool passive = false;

    //! The byte resolution
    unsigned byte_resolution = 1;

  private:

    /*
      The following methods and attributes are used only if CUDA is enabled
    */

    //! Load bytes from shared memory directory to GPU memory
    int64_t load_bytes_device (unsigned char* device_memory, uint64_t bytes, void * device_handle) override;

    //! Require that shared memory is lockable and registerable with the CUDA driver
    void set_require_registered_memory() { require_registered_memory = true; };

    //! Zero the input data_block after reading values
    unsigned zero_input = 0;

    //! device buffer containing zeros
    void* zeroed_buffer = nullptr;

    //! size of the zeroed buffer
    uint64_t zeroed_buffer_size = 0;

    //! flag to control if shared memory must locked and registered with the CUDA driver
    bool require_registered_memory = false;

    //! flag to track whether the shared memory has been locked and registered with the CUDA driver
    bool registered_memory = false;

  };

}

#endif // !defined(__DADABuffer_h)
