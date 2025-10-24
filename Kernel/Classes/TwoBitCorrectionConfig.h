//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2021 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/TwoBitCorrectionConfig.h

#ifndef __TwoBitCorrectionConfig_h
#define __TwoBitCorrectionConfig_h

#include "dsp/TwoBitCorrection.h"

namespace dsp
{
  class TwoBitCorrection::Config
  {
  public:

    Config ();

    //! Configure the Input if it is a TwoBitCorrection
    void configure (Unpacker *);

    //! Return a help string
    std::string help (const std::string& arg);

    //! Parse a string
    std::string parse (const std::string& text);
    
    // when unpacking 2-bit data, apply dynamic output level setting
    void set_dynamic_output_level_setting (bool flag)
    {  dynamic_output_level_setting = flag; }
    bool get_dynamic_output_level_setting () const
    { return dynamic_output_level_setting; }

    // number of time samples used to estimate undigitized power
    void set_excision_nsample (unsigned nsample) 
    { excision_nsample = nsample; }
    unsigned get_excision_nsample () const
    { return excision_nsample; }

    // cutoff power used for impulsive interference rejection
    void set_excision_cutoff (float cutoff) 
    { excision_cutoff = cutoff; }
    float get_excision_cutoff () const { return excision_cutoff; }

    // sampling threshold
    void set_excision_threshold (float threshold)
    { excision_threshold = threshold; }
    float get_excision_threshold () const { return excision_threshold; }

  protected:
    
    // when unpacking 2-bit data, apply dynamic output level setting
    bool dynamic_output_level_setting;

    // number of time samples used to estimate undigitized power
    unsigned excision_nsample;

    // cutoff power used for impulsive interference rejection
    float excision_cutoff;

    // sampling threshold
    float excision_threshold;
  };

}

#endif
