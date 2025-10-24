/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Andrew Jameson, Willem van Straten and Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToQuantize.h"
#include "dsp/RescaleMedianMadCalculator.h"
#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/WeightedTimeSeries.h"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_CUDA
#include "dsp/ChanPolSelectCUDA.h"
#include "dsp/GenericVoltageDigitizerCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "dsp/RescaleCUDA.h"
#include "dsp/RescaleMedianMadCalculatorCUDA.h"
#include "dsp/RescaleMeanStdDevCalculatorCUDA.h"
#include "dsp/TransferBitSeriesCUDA.h"
#include "dsp/TimeSeriesCUDA.h"
#endif

using namespace std;

dsp::LoadToQuantize::Config::Config()
{
#ifdef HAVE_CUDA
  can_cuda = true;
#endif
}

dsp::LoadToQuantize::Config::Config(const std::string& _channel_range, const std::string& _pol_range, unsigned _output_nbit)
{
  channel_range = _channel_range;
  pol_range = _pol_range;
  output_nbit = _output_nbit;

#ifdef HAVE_CUDA
  can_cuda = true;
#endif
}

dsp::LoadToQuantize::LoadToQuantize(Sink<BitSeries> * sink, Config* configuration) try
{
  if (configuration)
    set_configuration(configuration);
  output = sink;
}
catch(Error& error)
{
  throw error += "LoadToQuantize ctor Config*=" + tostring((void*)configuration);
}

void dsp::LoadToQuantize::set_configuration(Config* configuration)
{
  SingleThread::set_configuration(configuration);
  config = configuration;
}

void dsp::LoadToQuantize::construct() try
{
  SingleThread::construct();

  // get the time series output by the Source
  Reference::To<TimeSeries> to_process = SingleThread::source_output;
#ifdef HAVE_CUDA
  bool run_on_gpu = thread_id < config->get_cuda_ndevice();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>( gpu_stream );

  if (run_on_gpu)
  {
    if (dsp::Operation::verbose)
    {
      cerr << "dsp::LoadToQuantize::construct creating new CUDA::TimeSeriesEngine" << endl;
    }
    to_process->set_memory (device_memory);
    to_process->set_engine (new CUDA::TimeSeriesEngine (device_memory));

    // ensure the weights are processed on the GPU
    auto to_process_wts = dynamic_cast<WeightedTimeSeries*>(to_process.get());
    if (to_process_wts)
    {
      to_process_wts->set_weights_memory(device_memory);
    }
  }
#else
  // default to running on CPU if CUDA not available
  bool run_on_gpu = false;
#endif

  Observation * obs = get_source()->get_info();
  const unsigned nchan = obs->get_nchan();
  const unsigned npol = obs->get_npol();

  auto dada_header = dynamic_cast<::dsp::ASCIIObservation*>(obs);
  if (dada_header)
  {
    /*
      Record the original dimensions of the data before decimation, for use in offline analysis
    */
    dada_header->append("ORIGINAL_NCHAN", obs->get_nchan());
    dada_header->append("ORIGINAL_NPOL", obs->get_npol());
    dada_header->append("ORIGINAL_NBIT", obs->get_nbit());
  }

  /* Here is where the pipeline is created and configured */
  std::pair<uint32_t, uint32_t> channels = config->parse_range(config->channel_range, nchan);
  std::pair<uint32_t, uint32_t> pols = config->parse_range(config->pol_range, npol);

  output_nchan = (channels.second - channels.first) + 1;
  output_npol = (pols.second - pols.first) + 1;
  output_nbit = config->output_nbit;

  if (dsp::Operation::verbose)
    cerr << "dsp::LoadToQuantize::construct output nchan=" << output_nchan << " npol=" << output_npol
      << " nbit=" << output_nbit << endl;

  switch (output_nbit)
  {
  case 1:
  case 2:
  case 4:
  case 8:
  case 16:
    break;
  default:
    throw Error(InvalidParam, "dsp::LoadToQuantize::construct", "invalid output nbit [%u] not in [1, 2, 4, 8, 16]", output_nbit);
  }

  if (output_nchan < nchan || output_npol < npol)
  {
    chan_pol_select = new dsp::ChanPolSelect;
#ifdef HAVE_CUDA
    if (run_on_gpu)
    {
      if (dsp::Operation::verbose)
      {
        cerr << "dsp::LoadToQuantize::construct creating new CUDA::ChanPolSelectEngine" << endl;
      }
      chan_pol_select->set_engine (new CUDA::ChanPolSelectEngine (stream));
    }
#endif

    chan_pol_select->set_start_channel_index(channels.first);
    chan_pol_select->set_number_of_channels_to_keep(output_nchan);
    chan_pol_select->set_start_polarization_index(pols.first);
    chan_pol_select->set_number_of_polarizations_to_keep(output_npol);

    // the chan_pol_select operation is performed out-of-place
    chan_pol_select->set_input(to_process);
    to_process = new_time_series();
#ifdef HAVE_CUDA
    if (run_on_gpu)
    {
      to_process->set_memory (device_memory);
      auto to_process_wts = dynamic_cast<WeightedTimeSeries*>(to_process.get());
      if (to_process_wts)
      {
        to_process_wts->set_weights_memory(device_memory);
      }
    }
#endif
    chan_pol_select->set_output(to_process);
    operations.push_back(chan_pol_select.get());
  }

  // the rescale operation is performed in-place
  rescale = new dsp::Rescale();
  if (!run_on_gpu)
  {
    if (config->use_median_mad)
    {
      rescale->set_calculator(new RescaleMedianMadCalculator);
    }
  }
#ifdef HAVE_CUDA
  else
  {
    if (dsp::Operation::verbose)
    {
      cerr << "dsp::LoadToQuantize::construct creating new CUDA::RescaleEngine" << endl;
    }
    if (config->use_median_mad)
    {
      rescale->set_calculator(new CUDA::RescaleMedianMadCalculatorCUDA (stream));
    }
    else
    {
      rescale->set_calculator(new CUDA::RescaleMeanStdDevCalculatorCUDA (stream));
    }
    rescale->set_engine (new CUDA::RescaleEngine (stream));
  }
#endif
  rescale->set_input(to_process);
  rescale->set_output(to_process);
  rescale->set_constant(config->rescale_constant);
  if (config->rescale_interval > 0.0)
  {
    rescale->set_interval_seconds(config->rescale_interval);
  }

  if (!config->scale_offset_filename.empty())
  {
    scale_offset_dump = new dsp::RescaleScaleOffsetDump(config->scale_offset_filename);
    if (dada_header)
    {
      if (Operation::verbose)
      {
        cerr << "dsp::LoadToQuantize::construct copying DADA header from source to RescaleScaleOffsetDump" << endl;
        cerr << "dsp::LoadToQuantize::construct ASCII header=" << dada_header->get_header() << endl;
      }
      scale_offset_dump->get_header()->set_header(dada_header->get_header());
    }

    rescale->add_callback_handler(scale_offset_dump, &dsp::RescaleScaleOffsetDump::handle_scale_offset_updated);
  }

  operations.push_back(rescale.get());

  // the output
  quantized = new dsp::BitSeries();

  // the digitizer operation is performed out-of-place
  digitizer = new dsp::GenericVoltageDigitizer();
#ifdef HAVE_CUDA
  if (run_on_gpu)
  {
    if (dsp::Operation::verbose)
    {
      cerr << "dsp::LoadToQuantize::construct ensuring digitizer uses CUDA::DeviceMemory" << endl;
    }
    digitizer->set_device(device_memory);
    quantized->set_memory(device_memory);
  }
#endif

  digitizer->set_nbit(output_nbit);
  digitizer->set_input(to_process);
  digitizer->set_output(quantized);
  operations.push_back(digitizer.get());

#ifdef HAVE_CUDA
  if (run_on_gpu)
  {
    TransferBitSeriesCUDA* transfer = new TransferBitSeriesCUDA (stream);
    transfer->set_kind (cudaMemcpyDeviceToHost);
    transfer->set_input( quantized );
    transfer->set_output( quantized = new BitSeries() );
    quantized->set_memory( new CUDA::PinnedMemory() );
    operations.push_back (transfer);
  }
#endif

  // the output data sink
  output->set_input(quantized);
  operations.push_back(output.get());

  auto weighted = dynamic_cast<WeightedTimeSeries*>(to_process.get());
  if (weighted && output_weights)
  {
    quantized_weights = new dsp::BitSeries();
    digitizer->set_output_weights(quantized_weights);
#ifdef HAVE_CUDA
    if (run_on_gpu)
    {
      quantized_weights->set_memory(device_memory);

      TransferBitSeriesCUDA* transfer = new TransferBitSeriesCUDA (stream);
      transfer->set_kind (cudaMemcpyDeviceToHost);
      transfer->set_input( quantized_weights );
      transfer->set_output( quantized_weights = new BitSeries() );
      quantized_weights->set_memory( new CUDA::PinnedMemory() );
      operations.push_back (transfer);
    }
#endif
    output_weights->set_input(quantized_weights);
    operations.push_back(output_weights.get());
  }
}
catch(Error& error)
{
  throw error += "dsp::LoadToQuantize::construct";
}

void dsp::LoadToQuantize::Config::add_options(CommandLine::Menu& menu) try
{
  SingleThread::Config::add_options(menu);

  auto arg = menu.add(output_nbit, 'b', "bits");
  arg->set_help("number of bits per output sample (default: 8, supported: [8])");

  arg = menu.add(channel_range, "outchan", "output_channels");
  arg->set_help("output channel range in form first:last (default first:last input channel)");

  arg = menu.add(pol_range, "outpol", "output_pols");
  arg->set_help("output polarisation range in form first:last (default first:last input polarisation)");

  arg = menu.add (rescale_constant, 'c');
  arg->set_help ("keep offset and scale constant");

  arg = menu.add (rescale_interval, 'I', "secs");
  arg->set_help ("rescale interval in seconds");

  arg = menu.add (use_median_mad, "mad");
  arg->set_help ("use median/MAD for rescaling to mitigate effects of RFI");

  arg = menu.add (scale_offset_filename, "scloffsout", "scale_offset_file");
  arg->set_help ("output scales and offsets filename");
}
catch(Error &error)
{
  throw error += "dsp::LoadToQuantize::add_options";
}

std::pair<uint32_t, uint32_t> dsp::LoadToQuantize::Config::parse_range(const std::string &range, uint32_t max_range)
{
  std::pair<uint32_t, uint32_t> default_range(0, max_range - 1);

  if (range == "")
  {
    return default_range;
  }

  std::istringstream iss(range);
  std::string first;
  std::string second;

  if (!std::getline(iss, first, ':'))
  {
    throw Error(InvalidParam, "dsp::LoadToQuantize::Config::parse_range", "failed to parse '%s' as pair of colon delimited integers", range.c_str());
  }
  if (!std::getline(iss, second, ':'))
  {
    throw Error(InvalidParam, "dsp::LoadToQuantize::Config::parse_range", "failed to parse '%s' as pair of colon delimited integers", range.c_str());
  }

  uint32_t parsed_first = atoi(first.c_str());
  uint32_t parsed_second = atoi(second.c_str());
  if (parsed_first > parsed_second)
  {
    throw Error(InvalidParam, "dsp::LoadToQuantize::Config::parse_range", "upper index [%u] was greater than lower index [%u]", parsed_first, parsed_second);
  }
  if (parsed_second >= max_range)
  {
    throw Error(InvalidParam, "dsp::LoadToQuantize::Config::parse_range", "upper index [%u] was greater than maximum index [%u]", parsed_second, max_range);
  }
  return std::pair<uint32_t, uint32_t>(parsed_first, parsed_second);
}

void dsp::LoadToQuantize::prepare() try
{
  SingleThread::prepare();

  // If the input/source does not produce weights, then remove the output_weights step
  if (output_weights && !digitizer->has_output_weights())
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToQuantize::prepare digitizer has no output weights - deleting output_weights operation" << endl;
    assert(operations.back() == output_weights);
    operations.pop_back();
    output_weights = nullptr;
  }
}
catch(Error &error)
{
  throw error += "dsp::LoadToQuantize::prepare";
}

void dsp::LoadToQuantize::finish() try
{
  SingleThread::finish();

  /* anything extra */
}
catch(Error &error)
{
  throw error += "dsp::LoadToQuantize::finish";
}

