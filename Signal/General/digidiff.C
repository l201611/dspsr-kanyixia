/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  digidiff - reports on the difference between two files
 */

#include "dsp/SourceFactory.h"
#include "dsp/WeightedTimeSeries.h"

#include "CommandLine.h"
#include "dirutil.h"

using namespace std;

// print the reduced chi squared of the difference
bool print_reduced_chisq = false;

void parse_options (int argc, char** argv) try
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;
  
  menu.set_help_header ("digidiff - convert input to floating-point DADA file segments");
  menu.set_version ("digidiff " + tostring(dsp::version));

  arg = menu.add (print_reduced_chisq, 'X');
  arg->set_help ("print the reduced chi squared of the difference");

  dsp::Verbosity verbosity;
  arg = menu.add (&verbosity, &dsp::Verbosity::increase, 'v');
  arg->set_help ("increase verbosity");

  menu.parse (argc, argv);
}
catch (Error& error)
{
  cerr << error << endl;
  exit (-1);
}
catch (std::exception& error)
{
  cerr << error.what() << endl;
  exit (-1);
}

int diff_two (const std::string& fileA, const std::string& fileB)
{
  dsp::SourceFactory factory;
  Reference::To<dsp::Source> sourceA = factory.create(fileA);
  Reference::To<dsp::Source> sourceB = factory.create(fileB);

  auto obsA = sourceA->get_info();
  auto obsB = sourceB->get_info();

  // the Observation::combinable method compares a bunch of attributes like nchan, npol, freq, bw, etc.
  if (! obsA->combinable(obsB))
  {
    cerr << "digidiff: input files are not compatible " << obsA->get_reason() << endl;
    return -1;
  }

  if (sourceA->get_total_samples() != sourceB->get_total_samples())
  {
    cerr << "digidiff: input files contain different numbers of samples " << sourceA->get_total_samples() << " != " << sourceB->get_total_samples() << endl;
    return -1;
  }

  unsigned nchan = obsA->get_nchan();
  unsigned npol = obsA->get_npol();
  unsigned ndim = obsA->get_ndim();

  unsigned ndat = 1024;
  sourceA->set_block_size(ndat);
  sourceB->set_block_size(ndat);
  
  dsp::WeightedTimeSeries timeseriesA;
  dsp::WeightedTimeSeries timeseriesB;

  sourceA->set_output(&timeseriesA);
  sourceB->set_output(&timeseriesB);

  double total_diffsq = 0.0;
  uint64_t total_count = 0;

  while (! (sourceA->end_of_data() || sourceB->end_of_data()) )
  {
    sourceA->operate();
    sourceB->operate();

    unsigned nfloat = ndat * ndim;

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        const float* dataA = timeseriesA.get_datptr (ichan, ipol);
        const float* dataB = timeseriesB.get_datptr (ichan, ipol);

        for (unsigned ifloat=0; ifloat < nfloat; ifloat++)
        {
          double diff = dataA[ifloat] - dataB[ifloat];
          total_diffsq += diff*diff;
          total_count ++;
        }
      }
    }
  }

  if (sourceA->end_of_data() != sourceB->end_of_data())
  {
    cerr << "digidiff: input files reached different ends of data " << sourceA->end_of_data() << " != " << sourceB->end_of_data() << endl;
    return -1;
  }

  double variance = total_diffsq / total_count;
  cout << sqrt(variance) << endl;
  return 0;
}

int main (int argc, char** argv) try
{
  parse_options (argc, argv);

  vector <string> filenames;
  for (int ai=optind; ai<argc; ai++)
    dirglob (&filenames, argv[ai]);

  if (filenames.size() != 2)
  {
    cerr << "digidiff: please specify the names of two files to be diffed" << endl;
    return -1;
  }

  return diff_two (filenames[0], filenames[1]);
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}
