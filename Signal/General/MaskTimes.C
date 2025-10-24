/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/MaskTimes.h"
#include "Expect.h"
#include "strutil.h"

using namespace std;

dsp::MaskTimes::MaskTimes () : Mask("MaskTimes")
{
  if (verbose)
    cerr << "dsp::MaskTimes::MaskTimes()" << endl;
  current_interval = 0;
}

dsp::MaskTimes::~MaskTimes ()
{
  if (verbose)
    cerr << "dsp::MaskTimes::~MaskTimes()" << endl;
}

void dsp::MaskTimes::mask_data ()
{
  if (verbose)
    cerr << "dsp::MaskTimes::mask_data current_interval=" << current_interval << endl;
  
  uint64_t ndat = input->get_ndat();
  double rate = input->get_rate();
  double duration = ndat / rate;

  TimeInterval data_interval(input->get_start_time(),duration);

  while (current_interval > 0 && interval[current_interval].after(data_interval))
    current_interval --;

  while (current_interval < interval.size() && interval[current_interval].before(data_interval))
    current_interval ++;

  while (current_interval < interval.size() && interval[current_interval].overlaps(data_interval))
  {
    double time_offset = (interval[current_interval].start_time-data_interval.start_time).in_seconds();
    uint64_t start_offset = 0;
    if (time_offset > 0)
      start_offset = time_offset * rate;
    
    time_offset = (interval[current_interval].end_time-data_interval.start_time).in_seconds();
    uint64_t end_offset = time_offset * rate;
    if (end_offset > ndat)
      end_offset = ndat;

    current_interval ++;

    if (verbose)
      cerr << "dsp::MaskTimes::mask_data start_offset=" << start_offset << " end_offset=" << end_offset << endl;

    mask (start_offset, end_offset);
  }
}

void dsp::MaskTimes::load (const std::string& filename)
{
  Expect test (filename);
  string expect = "# MJD duration";

  if (!test.expect(expect))
    throw Error (InvalidParam, "dsp::MaskTimes::load", filename+" does not start with '"+expect+"'");

  std::vector<std::string> lines;
  loadlines(filename, lines);

  if (verbose)
    cerr << "dsp::MaskTimes::load " << lines.size() << " lines loaded" << endl;

  const string whitespace = " \t\n";

  for (unsigned i=0; i<lines.size(); i++)
  {
    string time = lines[i];
    string start = stringtok(time,whitespace);
    MJD start_time(start);
    double duration = fromstring<double> (time);

    if (verbose)
      cerr << "\t line " << i << " txt='" << lines[i] << "' MJD=" << start_time << " T=" << duration << endl;

    interval.push_back(TimeInterval(start_time, duration));
  }
}
