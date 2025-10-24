//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Formats/cpsr2/dsp/CPSR2TwoBitCorrection.h

#ifndef __CPSR2TwoBitCorrection_h
#define __CPSR2TwoBitCorrection_h

class CPSR2TwoBitCorrection;

#include "dsp/TwoBitCorrection.h"

namespace dsp {

  class TwoBitTable;

  /*
   * 2023-Jun-29 - Willem van Straten
   *
   * The CPSR2 instrument was decommissioned in 2010 and it is
   * not recommended to begin any new development using this software.
   * See the warning in CSPR2File.h for more information.
   *
   */

  //! Converts CPSR2 data from 2-bit digitized to floating point values
  class CPSR2TwoBitCorrection: public TwoBitCorrection {

  public:

    //! Constructor initializes base class attributes
    CPSR2TwoBitCorrection ();

    //! Return true if CPSR2TwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif
