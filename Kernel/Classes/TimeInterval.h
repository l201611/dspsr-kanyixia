//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/TimeInterval.h

#ifndef __dsp_Kernel_Classes_TimeInterval_h
#define __dsp_Kernel_Classes_TimeInterval_h

#include "MJD.h"

//! A time TimeInterval
class TimeInterval
{
  public:
  MJD start_time;
  MJD end_time;
  double duration;  // duration in seconds

  //! Construct from a start time and duration in seconds
  TimeInterval (const MJD& start, double seconds);

  //! Return true if this TimeInterval overlaps (partially spans) the given TimeInterval
  bool overlaps (const TimeInterval&);

  //! Return true if this TimeInterval ends before the given TimeInterval
  bool before (const TimeInterval&);

  //! Return true if this TimeInterval starts after the given TimeInterval
  bool after (const TimeInterval&);
};

#endif
