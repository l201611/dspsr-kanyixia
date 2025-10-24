//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/Digitizer.h


#ifndef __Digitizer_h
#define __Digitizer_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

namespace dsp {

  //! Convert floating point samples to N-bit samples
  class Digitizer : public Transformation <TimeSeries, BitSeries>
  {

  public:
    
    //! Constructor
    Digitizer (const char* name = "Digitizer");
    
    //! Set the number of bits per sample (FITS BITPIX convention)
    virtual void set_nbit (int);

    //! Get the number of bits per sample (FITS BITPIX convention)
    int get_nbit () const;

    //! Copy the input attributes to the output
    virtual void prepare ();

    //! Resize the output
    virtual void reserve ();
    
    //! Return true if the unpacker can operate on the specified device
    virtual bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    virtual void set_device (Memory*);

   protected:

    virtual void transformation ();
    
    //! Perform the digitization
    virtual void pack () = 0;

    //! Number of bits per sample (default = 32-bit float)
    int nbit = -32;

  };

}

#endif // !defined(__Digitizer_h)
