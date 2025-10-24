//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/ASCIIObservation.h

#ifndef __ASCIIObservation_h
#define __ASCIIObservation_h

#include "dsp/Observation.h"
#include "ascii_header.h"

namespace dsp {

  //! Parses Observation attributes from an ASCII header
  /*! This class parses the ASCII header block used by DADA-based
    instruments such as CPSR2, PuMa2, and APSR.  It initializes all of
    the attributes of the Observation base class.  The header block
    may come from a data file, or from shared memory. */
  class ASCIIObservation : public Observation
  {

  public:

    //! Construct from an ASCII header block
    ASCIIObservation (const char* header=0);

    //! Construct from an Observation
    ASCIIObservation (const Observation*);

    //! Cloner (calls new)
    ASCIIObservation* clone() const override;

    //! Read the ASCII header block
    void load (const char* header);

    //! Write an ASCII header block
    void unload (char* header);

    //! Get the number of bytes offset from the beginning of acquisition
    uint64_t get_offset_bytes () const { return offset_bytes; }

    //! Set/unset a required keyword
    void set_required (std::string key, bool required=true);

    //! Check if a certain keyword is required
    bool is_required (std::string key);

    template <typename T>
    int custom_header_get (std::string key, const char *format, T result) const;

    //! Return the immutable ASCII header from which this was constructed
    const char* get_header () const { return loaded_header.c_str(); }

    //! Return the mutable ASCII header from which this was constructed
    char* get_header () { return const_cast<char*>(loaded_header.c_str()); }

    //! Get the number of samples that must be loaded at one time
    unsigned get_resolution () const;

    /**
     * @brief Append a key/value pair to the header text
     *
     * @tparam T type of the value to set
     * @param key keyword that names the parameter
     * @param val value of the parameter
     */
    template <typename T>
    void append(const std::string &key, T val)
    {
      append_keyval(key, tostring(val));
    }

    /**
     * @brief Append a key/value pair to the header text
     *
     * @param key keyword that names the parameter
     * @param val string value of the parameter
     */
    void append_keyval(const std::string &key, const std::string &val);

  protected:

    std::string hdr_version;

    //! The list of ASCII keywords that must be present
    std::vector< std::string > required_keys;

    //! Load a keyword, only throw an error if it's required and doesn't exist
    template <typename T>
    int ascii_header_check (const char *header, std::string key, const char *format, T result);

    template <typename T>
    void load_str_into_array (std::string from, T* buffer, unsigned bufsize);

    //! Number of bytes offset from the beginning of acquisition
    uint64_t offset_bytes;

    std::string loaded_header;
  };

}

template <typename T>
void dsp::ASCIIObservation::load_str_into_array ( std::string from, T* buffer, unsigned bufsize )
{
  std::string val;
  std::string delimiter = ",";
  size_t pos=0;
  for (unsigned i=0; i<bufsize; i++) {
    pos = from.find(delimiter);
    val = from.substr(0, pos);
    buffer[i] = fromstring<T>(val);
    from.erase(0, pos + delimiter.length());
  }
}


template <typename T>
int dsp::ASCIIObservation::custom_header_get ( std::string key, const char *format, T result) const
{
  int rv = ascii_header_get (loaded_header.c_str(), key.c_str(), format, result);
  if ( rv > 0)
    return rv;
  throw Error (InvalidState, "ASCIIObservation::custom_header_get", "failed to find " + key);
}


template <typename T>
int dsp::ASCIIObservation::ascii_header_check (const char *header,
    std::string key, const char *format, T result)
{
  int rv = ascii_header_get(header, key.c_str(), format, result);

  if ( rv>0 || !is_required(key) )
    return rv;

  throw Error (InvalidState, "ASCIIObservation", "failed load " + key);
}

#endif
