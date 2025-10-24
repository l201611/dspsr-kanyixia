/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "TimeInterval.h"

TimeInterval::TimeInterval (const MJD& start, double T)
{
  start_time = start;
  duration = T;
  end_time = start + T;
}

bool TimeInterval::overlaps (const TimeInterval& that)
{
  return this->end_time > that.start_time && this->start_time < that.end_time;
}

bool TimeInterval::before (const TimeInterval& that)
{
  return this->end_time < that.start_time;
}

bool TimeInterval::after (const TimeInterval& that)
{
  return this->start_time > that.end_time;
}
