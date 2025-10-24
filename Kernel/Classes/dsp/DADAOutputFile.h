//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024 by Andrew Jameson & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/DADAOutputFile.h

#ifndef __dsp_DADAOutputFile_h
#define __dsp_DADAOutputFile_h

#include "dsp/OutputFile.h"
#include "dsp/DADAHeader.h"

namespace dsp {

  /**
   * @brief Writes BitSeries data to a single DADA File
   *
   */
  class DADAOutputFile : public OutputFile
  {
  public:

    /**
     * @brief Construct a new DADAOutputFile object
     *
     * @param filename if specified, path to the file to be opened
     */
    DADAOutputFile (const char* filename = nullptr);

    /**
     * @brief Construct a new DADAOutputFile object
     *
     * @param filename path to the file to be opened
     */
    DADAOutputFile (const std::string& filename);

    /**
     * @brief Get the DADA header manager
     * 
     */
    DADAHeader* get_header () { return &dada_header; }

  protected:

    /**
     * @brief Write the input BitSeries header to the open file, via dsp::ASCIIObservation::unload.
     *
     */
    void write_header();

    /**
     * @brief Get the extension to be added to the end of new filenames.
     *
     * @return std::string The extension to be used will be "dada"
     */
    std::string get_extension () const;

    //! Manages the ASCII header that will be written to the start of each file
    DADAHeader dada_header;
  };

} // namespace dsp

#endif // !defined(__dsp_DADAOutputFile_h)

