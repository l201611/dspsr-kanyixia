//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/cpsr2/dsp/CPSR2File.h

#ifndef __CPSR2File_h
#define __CPSR2File_h

#include "dsp/File.h"

namespace dsp {

  /*
   * 2023-Jun-29 - Willem van Straten
   *
   * The CPSR2 instrument was decommissioned in 2010 and it is
   * not recommended to begin any new development using this software.
   * See the warning in CSPR2File.h for more information.
   *
   * The CPSR2 instrument output 16-byte frame headers at the start of
   * every 2 MB block of data.  These frame headers were written *over* data 
   * and not inserted between blocks.   At the time, it was thought that 
   * this would have a negligible impact on integrated results, so the samples
   * over-written by these headers are neither flagged nor skipped.
   *
   * Owing to this feature, I would not recommend using the CPSR2 file format 
   * for new development.  There are other similarly simple file formats, like 
   * the DADA file format that is built in to dspsr.  See
   *
   * Kernel/Classes/DADAFile.C
   *
   * and
   *
   * Kernel/Classes/GenericEightBitUnpacker.C
   * Kernel/Classes/GenericFourBitUnpacker.C
   * Kernel/Classes/TwoBitCorrection.C
   *
   */

  //! Loads BitSeries data from a CPSR2 data file
  class CPSR2File : public File 
  {
  public:
   
    //! Construct and open file
    CPSR2File (const char* filename=0);

    virtual ~CPSR2File();

    //! Returns true if filename appears to name a valid CPSR2 file
    bool is_valid (const char* filename) const;

    //! Set this to 'false' if you don't need to yamasaki verify
    static bool want_to_yamasaki_verify;

    //! return 'm' for cpsr1 and 'n' for cpsr2
    std::string get_prefix () const;

  protected:

    //! Pads gaps in data
    virtual int64_t pad_bytes(unsigned char* buffer, int64_t bytes);
      
    //! Open the file
    virtual void open_file (const char* filename);

    //! Read the CPSR2 ascii header from filename
    static int get_header (char* cpsr2_header, const char* filename);

    std::string prefix;
  };

}

#endif // !defined(__CPSR2File_h)
  
