/***************************************************************************
 *
 *   Copyright (C) 2021 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/TwoBitCorrectionConfig.h"

using namespace std;
using namespace dsp;

TwoBitCorrection::Config::Config ()
{
  // by default, apply dynamic output level setting
  dynamic_output_level_setting = true;

  // number of time samples used to estimate undigitized power
  excision_nsample = 0;

  // cutoff power used for impulsive interference rejection
  excision_cutoff = -1.0;

  // sampling threshold
  excision_threshold = -1.0;
}

std::string TwoBitCorrection::Config::help (const std::string& arg)
{
  string p = " -" + arg;
  
  return
    p+ "c<cutoff>    threshold for impulsive interference excision \n" +
    p+ "n<sample>    number of samples used to estimate undigitized power \n" +
    p+ "t<threshold> two-bit sampling threshold at record time \n" +
    p+ "d            disable dynamic output level setting";
}

std::string TwoBitCorrection::Config::parse (const std::string& text)
{
  if (text == "d")
  {
    dynamic_output_level_setting = false;
    return "disabling dynamic output level setting";
  }

  const char* carg = text.c_str();

  int scanned = sscanf (carg, "n%u", &excision_nsample);
  if (scanned == 1)
    return "using " + tostring(excision_nsample) +
      " samples to estimate undigitized power";

  scanned = sscanf (carg, "c%f", &excision_cutoff);
  if (scanned == 1)
    return "setting impulsive interference excision threshold to "
      + tostring(excision_cutoff);

  scanned = sscanf (carg, "t%f", &excision_threshold);
  if (scanned == 1)
    return "setting two-bit sampling threshold to "
      + tostring(excision_threshold);

  throw Error (InvalidParam, "TwoBitCorrection::Config::parse",
	       "unrecognized text=" + text);
}

void TwoBitCorrection::Config::configure (Unpacker* unpacker)
{
  if (! dynamic_output_level_setting )
  {
    dsp::TwoBitCorrection* twobit;
    twobit = dynamic_cast<dsp::TwoBitCorrection*> ( unpacker );
    if (twobit)
      twobit->set_dynamic_output_level_setting (false);
  }

  dsp::ExcisionUnpacker* excision;
  excision = dynamic_cast<dsp::ExcisionUnpacker*> ( unpacker );

  if (excision)
  {
    if ( excision_nsample )
      excision -> set_ndat_per_weight ( excision_nsample );

    if ( excision_threshold > 0 )
      excision -> set_threshold ( excision_threshold );

    if ( excision_cutoff >= 0 )
      excision -> set_cutoff_sigma ( excision_cutoff );
  }
}
