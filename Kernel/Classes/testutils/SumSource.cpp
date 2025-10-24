/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <dsp/SumSource.h>

dsp::test::SumSource::SumSource(unsigned _niterations) : Source("SumSource")
{
  niterations = _niterations;
}

void dsp::test::SumSource::operation()
{
  // need to zero out the data!
  output->multiply(0.0);
  
  for (auto& source : sources)
  {
    // ensure that all the sources have latest configuration and correct output size.
    source->get_output()->copy_configuration(output);
    source->get_output()->resize(output->get_ndat());
    source->operate();
    output->add(source->get_output());
  }

  iterations++;
  update_end_of_data();
}

dsp::Source* dsp::test::SumSource::clone() const
{
  auto clone = new dsp::test::SumSource();
  clone->set_output(output);
  for (auto& source : sources)
  {
    clone->add_source(source->clone());
  }

  return clone;
}

void dsp::test::SumSource::set_output(dsp::TimeSeries * _output)
{
  output = _output;
  info = _output;

  for (auto& source : sources)
  {
    auto curr_output = source->get_output();
    curr_output->copy_configuration(output);
    curr_output->resize(output->get_ndat());
  }

  set_output_order(_output->get_order());
  total_samples = _output->get_ndat();
}

void dsp::test::SumSource::seek_time(double second)
{
  for (auto& source : sources)
  {
    source->seek_time(second);
  }
}

void dsp::test::SumSource::set_total_time(double _second) {
  for (auto& source : sources)
  {
    source->set_total_time(second);
  }
  second = _second;
}

void dsp::test::SumSource::restart() {
  for (auto& source : sources)
  {
    source->restart();
  }
}

void dsp::test::SumSource::close() {
  for (auto& source : sources)
  {
    source->close();
  }
}

void dsp::test::SumSource::set_block_size(uint64_t _block_size) {
  for (auto& source : sources)
  {
    source->set_block_size(_block_size);
  }
  block_size = _block_size;
}

void dsp::test::SumSource::set_overlap(uint64_t _overlap) {
  for (auto& source : sources)
  {
    source->set_overlap(_overlap);
  }
  overlap = _overlap;
}

bool dsp::test::SumSource::get_order_supported(TimeSeries::Order order) const {
  for (auto& source : sources)
  {
    if (!source->get_order_supported(order))
      return false;
  }
  return true;
}

void dsp::test::SumSource::set_output_order(TimeSeries::Order _order) {
  for (auto& source : sources)
  {
    source->set_output_order(_order);
  }
  output_order = _order;
}

bool dsp::test::SumSource::get_device_supported(dsp::Memory* _device_memory) const {
  for (auto& source : sources)
  {
    if (!source->get_device_supported(_device_memory))
      return false;
  }
  return true;
}

void dsp::test::SumSource::set_device(dsp::Memory* _device_memory) {
  for (auto& source : sources)
  {
    source->set_device(_device_memory);
  }
  memory = _device_memory;
}

void dsp::test::SumSource::share(Source* source) {
  for (auto& source : sources)
  {
    source->share(source);
  }
}

void dsp::test::SumSource::set_context(ThreadContext* context) {
  for (auto& source : sources)
  {
    source->set_context(context);
  }
}

void dsp::test::SumSource::add_source(dsp::Source * _source)
{
  // we need a separate TimeSeries
  auto timeseries = new TimeSeries;
  timeseries->set_order(output_order);
  timeseries->copy_configuration(output);
  timeseries->resize(output->get_ndat());

  _source->set_output(timeseries);
  _source->set_output_order(output_order);

  sources.push_back(_source);
}

void dsp::test::SumSource::update_end_of_data()
{
  bool _eod = iterations >= niterations;
  for (auto& source : sources)
  {
    // if any source is EOD then the sum source is also EOD
    _eod &= source->end_of_data();
  }
  eod = _eod;
}
