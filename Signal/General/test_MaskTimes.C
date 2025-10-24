/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MaskTimes.h"

using namespace std;
using namespace dsp;

int main(int argc, char ** argv) try
{
  if (argc < 2)
  {
    cerr << "USAGE: " << argv[0] << " filename" << endl;
    return -1;
  }

  MaskTimes::verbose = true;
  MaskTimes times;
  times.load(argv[1]);

  return 0;
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

