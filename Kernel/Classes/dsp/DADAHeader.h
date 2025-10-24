//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/DADAHeader.h

#ifndef __dsp_DADAHeader_h
#define __dsp_DADAHeader_h

#include <vector>
#include <cstddef>

namespace dsp {

  /**
   * @brief Manages a DADA ASCII header block
   *
   */
  class DADAHeader
  {
  public:

    /**
     * @brief The default DADA ASCII header block size
     * 
     */
    static const unsigned default_header_size;

    /**
     * @brief Set the DADA header
     * 
     * @param ascii_header block of ASCII text that will be copied
     */
    void set_header(const char* ascii_header);

    /**
     * @brief Load the DADA header from filename
     * 
     * @param filename name of file from which DADA header will be loaded
     * If the file does not contain a DADA header, then an attempt is made to
     * load the header from an adjacent header file, where "adjacent" is defined
     * by replacing the extension with .hdr; e.g. filename.dat -> filename.hdr
     */
    void load_header (const char* filename);

    /**
     * @brief Get the immutable DADA header
     * 
     */
    const char* get_header() const { return header.data(); }

    /**
     * @brief Get the mutable DADA header
     * 
     */
    char* get_header() { return header.data(); }

    /**
     * @brief resize the header
     *
     * @param size desired header size
     * 
     * If size is less than default_header_size, it will be set to default_header_size
     * If size is greater than default_header_size, it will be set to the smallest power of two that is larger than size
     */
    void resize(unsigned size = 0);

    /**
     * @brief Get the current header size
     * 
     */
    size_t size() const { return header.size(); }

    /**
     * @brief Return true if the header is empty
     * 
     */
    bool empty () const;

    /**
     * @brief Return true if the header is valid
     * 
     */
    bool valid (bool verbose) const;
    
    /**
     * @brief Return the size of the header in the data file
     * 
     */
    unsigned get_header_size () const { return hdr_size; }

  protected:

    //! The ASCII header block
    mutable std::vector<char> header;

    //! The header is in a adjacent file
    bool adjacent_header_file = false;

    //! The size of the header in the data file
    unsigned hdr_size = 0;
  };

} // namespace dsp

#endif // !defined(__dsp_DADAHeader_h)
