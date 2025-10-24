/***************************************************************************
 *                                                                         *
 *   Copyright (C) 2024 by Jesmigel A. Cantos                              *
 *   Licensed under the Academic Free License version 2.1                  *
 *                                                                         *
 ***************************************************************************/

#ifndef __dsp_OutputDADABuffer_h
#define __dsp_OutputDADABuffer_h

#include "dsp/Sink.h"
#include "dsp/BitSeries.h"
#include "dsp/DADAHeader.h"

#include "dada_hdu.h"

#include <ipcio.h>
#include <ipcbuf.h>

namespace dsp {

  /**
   * @class OutputDADABuffer
   * @brief A class for writing BitSeries data to a DADA ring buffer.
   */
  class OutputDADABuffer : public Sink<BitSeries>
  {
    public:
      /**
       * @brief Constructor class for writing BitSeries data to the DADA ring buffer.
       * @param key_string The key for accessing the DADA ring buffer.
       */
      OutputDADABuffer(const std::string &key_string);

      /**
       * @brief Destructor.
       */
      ~OutputDADABuffer();

      /**
       * @brief Write BitSeries data to shared memory.
       */
      void calculation ();

      /**
       * @brief Open the data ring buffer for writing
       */
      void reopen();

      /**
       * @brief Write end-of-data to the data ring buffer and close it
       */
      void close();

      /**
       * @brief Get the DADA header manager
       *
       */
      DADAHeader* get_header () { return &dada_header; }

    private:
      /**
       * @brief Connect to the DADA ring buffer.
       * @param smrb_key The key for accessing the DADA ring buffer.
       */
      void connect(key_t smrb_key);

      /**
       * @brief Disconnect from the DADA ring buffer.
       */
      void disconnect();

      /**
       * @brief Write header information to the DADA ring buffer.
       */
      void write_header();

      /**
       * @brief Unload bytes to the DADA ring buffer.
       * @param buffer Pointer to the data buffer.
       * @param nbytes Number of bytes to unload.
       */
      void unload_bytes(const void* buffer, uint64_t nbytes);

      /**
       * @brief Parse the PSRDADA key from a string.
       * @param key_string The PSRDADA key string.
       * @return The parsed key.
       */
      key_t parse_psrdada_key(const std::string &key_string);

    protected:

      //! Shared memory interface
      dada_hdu_t* hdu = nullptr;

      //! Flag indicating connection status.
      bool connected = false;

      //! Flag indicating lock status.
      bool locked = false;

      //! Header has been written to header block in shared memory
      bool header_written = false;

      //! Manages the ASCII header that will be written to the header block
      DADAHeader dada_header;

    private:

      /**
       * @brief Display contents of the input data.
       */
      void display_input_contents();

  };
} // namespace dsp

#endif // __dsp_OutputDADABuffer_h
