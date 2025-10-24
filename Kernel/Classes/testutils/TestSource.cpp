
/***************************************************************************
 *
 *   Copyright (C) 2024 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <dsp/TestSource.h>
#include "dsp/OperationPerformanceMetrics.h"

dsp::test::TestSource::TestSource(unsigned _niterations) : TestSource("TestSource", _niterations) {}

dsp::test::TestSource::TestSource(const char* name, unsigned _niterations) : Source(name)
{
  niterations = _niterations;
}

void dsp::test::TestSource::operation()
{
  iterations++;
  bool done = (iterations >= niterations);
  set_end_of_data(done);
  performance_metrics->update_metrics(this->output);
}

dsp::Source* dsp::test::TestSource::clone() const
{
  TestSource * clone = new dsp::test::TestSource();
  clone->info = info;
  clone->output = output;
  return clone;
}

void dsp::test::TestSource::set_output(dsp::TimeSeries * _output)
{
  output = _output;
  info = _output;

  if (output->get_rate() <= 0)
  {
    output->set_rate(1.0);
  }

  set_total_samples(output->get_ndat());
}

void dsp::test::TestSource::set_output_order(dsp::TimeSeries::Order _order)
{
  output_order = _order;
  if (output) {
    output->set_order(_order);
  }
}
