//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Generators/dsp/ImpulseTrain.h

#ifndef __dsp_Signal_Generators_ImpulseTrain_h
#define __dsp_Signal_Generators_ImpulseTrain_h

#include "dsp/Generator.h"

namespace dsp {

  //! A train of delta functions, also known as a shah function or Dirac comb
  /*!
    And I didn't have any idea what to do
    But I knew I needed a click
  */
  class ImpulseTrain : public Generator
  {
  protected:

    //! Number of time samples between impulses
    uint64_t period = 0;
   
    //! Time sample index of next impulse
    uint64_t next = 0;

    //! Amplitude of each impulse
    float amplitude = 1.0;

  public:

    //! Default constructor
    ImpulseTrain ();

    //! Return a copy-constructed clone
    ImpulseTrain* clone() const override { return new ImpulseTrain(*this); }

    //! Fill the output TimeSeries with an impulse train
    void operation () override;
    
    //! Set the number of time samples between impulses
    void set_period_samples (uint64_t n) { period = n; }

    //! Set the index of the next impulse
    void set_next_sample (uint64_t n) { next = n; }

    //! Set the amplitude of each impulse
    void set_amplitude (float amp) { amplitude = amp; }
  };

}

#endif // !defined(__dsp_Signal_Generators_ImpulseTrain_h)
