/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToSlice.h"
#include "dsp/Source.h"

#include "dsp/GenericVoltageDigitizer.h"
#include "dsp/DADAOutputFile.h"

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::LoadToSlice::LoadToSlice (Config* configuration) try
{
  if (configuration)
    set_configuration (configuration);
}
catch (Error& error)
{
  throw error += "LoadToSlice ctor Config*=" + tostring ((void*)configuration);
}

//! Run through the data
void dsp::LoadToSlice::set_configuration (Config* configuration)
{
  SingleThread::set_configuration (configuration);
  config = configuration;
}

void dsp::LoadToSlice::construct () try
{
  SingleThread::construct ();

  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  // set up for optimal memory usage pattern
  Reference::To<TimeSeries> to_process = SingleThread::source_output;

  if (source->get_info()->get_detected())
  {
    if (config->dedisp.coherent_derotation)
      throw Error (InvalidParam, "dsp::LoadToSlice::construct",
                   "cannot perform phase-coherent Faraday rotation correction\n\t"
                   "because input signal is detected");

    // detected data is handled much more efficiently in TFP order
    if (config->optimal_order && source->get_order_supported(TimeSeries::OrderTFP))
    {
      source->set_output_order(TimeSeries::OrderTFP);
    }

    config->dedisp.coherent_dedispersion = false;

    to_process = construct_interchan(to_process);
  }
  else
  {
    config->dedisp.integrate_passband = false;
    if (report_vitals())
    {
      config->dedisp.report_as = config->application_name;
    }
    
    DedispersionPipe::config = &(config->dedisp);
    DedispersionPipe::pipeline = this;

    to_process = DedispersionPipe::construct(to_process);
  }

  truncate = new Truncate;
  truncate->set_input(to_process);
  append(truncate);

  Reference::To<GenericVoltageDigitizer> digitizer = new GenericVoltageDigitizer;
  Reference::To<BitSeries> digitized = new BitSeries;

  digitizer->set_input(to_process) ;
  digitizer->set_output(digitized);
  digitizer->set_nbit(-32);  // 32-bit float
  append(digitizer);

  outputFile = new DADAOutputFile;
  outputFile->set_input (digitized);

  // nanosecond precision for now
  outputFile->set_fractional_second_decimal_places(9);
  append(outputFile);
}
catch (Error& error)
{
  throw error += "dsp::LoadToSlice::construct";
}

//! Add command line options
void dsp::LoadToSlice::Config::add_options (CommandLine::Menu& menu)
{
  SingleThread::Config::add_options(menu);
  dedisp.add_options(menu);
}

void dsp::LoadToSlice::prepare()
{
  if (Operation::verbose)
    cerr << "dsp::LoadToSlice::prepare DM=" << config->dedisp.dispersion_measure
         << " RM=" << config->dedisp.rotation_measure << endl;

  DedispersionPipe::prepare();
  SingleThread::prepare();
}

//! Set the block size during prepare method
void dsp::LoadToSlice::prepare_block_size()
{
  if (Operation::verbose)
    cerr << "LoadToSlice::prepare_block_size" << endl;
  config->dedisp.input_buffering = config->input_buffering; // kludge
  DedispersionPipe::prepare(this);
}

//! Set the start of the time slice
void dsp::LoadToSlice::set_start_time (const MJD& epoch)
{
  seek_epoch(epoch);
}

//! Set the end of the time slice
void dsp::LoadToSlice::set_end_time (const MJD& epoch) try
{
  truncate->set_end_time(epoch);
}
catch (Error& error)
{
  throw error += "dsp::LoadToSlice::set_end_time";
}

void dsp::LoadToSlice::finish() try
{
  outputFile->close();
}
catch (Error &error)
{
  throw error += "dsp::LoadToSlice::finish";
}
