/**************************************************************************2
 *
 *   Copyright (C) 2024-2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/DedispersionPipe.h"
#include "dsp/DedispersionPipeConfig.h"

#include "dsp/SingleThread.h"
#include "dsp/Source.h"
#include "dsp/WeightedTimeSeries.h"

#include "dsp/DedispersionSampleDelay.h"
#include "dsp/Derotation.h"

#include "dsp/DynamicFilter.h"
#include "dsp/ResponseProduct.h"
#include "dsp/Apodization.h"
#include "dsp/RFIFilter.h"
#include "dsp/PolnCalibration.h"
#include "dsp/Response.h"

#include "dsp/FIRFilter.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankResponse.h"
#include "dsp/Filterbank.h"
#include "dsp/FilterbankEngineCPU.h"
#include "dsp/OptimalFFT.h"
#include "dsp/SampleDelay.h"

#if HAVE_CUDA
#include "dsp/ConvolutionCUDA.h"
#include "dsp/ConvolutionCUDASpectral.h"
#include "dsp/OptimalFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/TimeSeriesCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "dsp/SampleDelayCUDA.h"
#endif

#include "Error.h"
#include "debug.h"

#include <assert.h>

using namespace std;

static void *const undefined_stream = (void *)-1;

dsp::DedispersionPipe::DedispersionPipe(Config *configuration)
try
{
  if (configuration)
    set_configuration(configuration);
}
catch (Error &error)
{
  throw error += "DedispersionPipe ctor";
}

dsp::DedispersionPipe::~DedispersionPipe()
{
}

void dsp::DedispersionPipe::set_configuration(Config *configuration)
try
{
  config = configuration;
}
catch (Error &error)
{
  throw error += "DedispersionPipe::set_configuration";
}

dsp::TimeSeries* dsp::DedispersionPipe::construct(dsp::TimeSeries* input) try
{
  if (Operation::verbose)
    cerr << "dsp::DedispersionPipe::construct" << endl;

  if (!input)
    throw Error (InvalidState, "dsp::DedispersionPipe::construct", "no input");

  if (!config)
    throw Error (InvalidState, "dsp::DedispersionPipe::construct", "no configuration");

  if (config->temporal_apodization_type != "")
  {
    temporal_apodization = new dsp::Apodization;
    temporal_apodization->set_type(config->temporal_apodization_type);
  }

  if (config->spectral_apodization_type != "")
  {
    spectral_apodization = new dsp::Apodization;
    spectral_apodization->set_type(config->spectral_apodization_type);
  }

  if (config->dispersion_measure == 0.0)
  {
    if (Operation::verbose)
      cerr << "dsp::DedispersionPipe::construct Disabling coherent dedispersion" << endl;
    config->coherent_dedispersion = false;
  }

  // the data are not detected, so set up phase coherent reduction path
  // NB that this does not necessarily mean coherent dedispersion.

  if (Operation::verbose)
    cerr << "dsp::DedispersionPipe::construct frequency response = nullptr" << endl;

  frequency_response = nullptr;
  
  if (config->coherent_dedispersion)
  {

    if (Operation::verbose)
      cerr << "dsp::DedispersionPipe::construct kernel = new Dedispersion" << endl;

    if (!dedisp)
      dedisp = new Dedispersion;

    frequency_response = dedisp;
  }
  
  if (config->coherent_derotation)
  {
    if (!derotate)
      derotate = new Derotation;

    append_response(frequency_response, derotate);
  }

  if (config->integrate_passband && !passband)
  {
    passband = new Response;
  }

  if (config->zap_rfi)
  {
    if (!rfi_filter)
      rfi_filter = new RFIFilter;

    rfi_filter->set_source(pipeline->get_source());
    append_response(frequency_response, rfi_filter);
  }

  if (config->dynamic_response_filename != "")
  {
    Reference::To<Pulsar::Archive> archive = Pulsar::Archive::load(config->dynamic_response_filename);
    auto dyn_resp = archive->get<Pulsar::DynamicResponse>();
    if (!dyn_resp)
      throw Error (InvalidParam, "dsp::DedispersionPipe::construct",
                "Pulsar::Archive with filename='" + config->dynamic_response_filename + " does not contain a Pulsar::DynamicResponse extension");

    dynamic_filter = new DynamicFilter(dyn_resp);
    append_response(frequency_response, dynamic_filter);
  }

  if (!config->calibrator_database_filename.empty())
  {
    auto polcal = new dsp::PolnCalibration;
    polcal->set_database_filename(config->calibrator_database_filename);
    append_response(frequency_response, polcal);
  }

  // convolved and filterbank are out of place
  Reference::To<TimeSeries> convolved = input;

  // by default, Convolution is performed after (or without) any (Inverse) Filterbank
  auto convolve_when = config->get_convolve_when();

  if (config->coherent_dedispersion && convolve_when == Filterbank::Config::Before)
  {
    convolved = construct_convolution(convolved);
  }

  if (config->inverse_filterbank_enabled())
  {
    convolved = construct_inverse_filterbank(convolved);
  }

  if (config->filterbank_enabled())
  {
    convolved = construct_filterbank(convolved);
  }

  if (config->coherent_dedispersion && convolve_when == Filterbank::Config::After)
  {
    convolved = construct_convolution(convolved);
  }

  convolved = construct_interchan(convolved);

  if (frequency_response)
  {
    string app = config->report_as;

    unsigned frequency_resolution = config->filterbank.get_freq_res();

    if (frequency_resolution)
    {
      if (!app.empty())
        cerr << app << ": setting filter length to " << frequency_resolution << endl;
      frequency_response->set_frequency_resolution(frequency_resolution);
    }

    if (config->times_minimum_nfft)
    {
      if (!app.empty())
        cerr << app << ": setting filter length to minimum times "
             << config->times_minimum_nfft << endl;
      frequency_response->set_times_minimum_nfft(config->times_minimum_nfft);
    }

    if (config->nsmear)
    {
      if (!app.empty())
        cerr << app << ": setting smearing to " << config->nsmear << endl;
      frequency_response->set_impulse_samples(config->nsmear);
    }

    if (config->use_fft_bench)
    {
      if (!app.empty())
        cerr << app << ": using benchmarks to choose optimal FFT length" << endl;

#if HAVE_CUDA
      if (pipeline->run_on_gpu())
        frequency_response->set_optimal_fft(new OptimalFilterbank("CUDA"));
      else
#endif
        frequency_response->set_optimal_fft(new OptimalFFT);
    }
  }

  return convolved.release();
}
catch (Error &error)
{
  throw error += "dsp::DedispersionPipe::construct";
}

dsp::TimeSeries* dsp::DedispersionPipe::construct_inverse_filterbank (TimeSeries* input)
{
  Filterbank::Config::When convolve_when = config->inverse_filterbank.get_convolve_when();

  Reference::To<TimeSeries> output = pipeline->new_time_series();

#if HAVE_CUDA
  if (pipeline->run_on_gpu())
  {
    output->set_memory(pipeline->get_memory());
  }
#endif

  config->inverse_filterbank.set_device(pipeline->get_memory());
  config->inverse_filterbank.set_stream(pipeline->get_gpu_stream());

  if (!inverse_filterbank)
  {
    inverse_filterbank = config->inverse_filterbank.create();
  }
  if (!pipeline->get_configuration()->input_buffering)
  {
    inverse_filterbank->set_buffering_policy(NULL);
  }

  inverse_filterbank->set_input(input);
  inverse_filterbank->set_output(output);

  if (temporal_apodization)
    inverse_filterbank->set_temporal_apodization(temporal_apodization);

  if (spectral_apodization)
    inverse_filterbank->set_spectral_apodization(spectral_apodization);

  // InverseFilterbank will always have a response.
  inverse_filterbank_response = new dsp::InverseFilterbankResponse;
  inverse_filterbank_response->set_apply_deripple(false);
  inverse_filterbank_response->set_input_overlap(config->inverse_filterbank.get_input_overlap());
  inverse_filterbank_response->set_pfb_dc_chan(pipeline->get_source()->get_info()->get_pfb_dc_chan());

  auto info = pipeline->get_source()->get_info();

  if (info->get_deripple_stages() > 0)
  {
    dsp::FIRFilter first_filter = info->get_deripple()[0];
#define SETTING_ALL_CHAN_IS_FIXED 0
#if SETTING_ALL_CHAN_IS_FIXED
    if (Operation::verbose)
      cerr << "DedispersionPipe::construct_inverse_filterbank FIRFilter::nchan=" << first_filter.get_pfb_nchan()
            << " Input::nchan=" << info->get_nchan() << endl;
    inverse_filterbank->set_pfb_all_chan(
        first_filter.get_pfb_nchan() == info->get_nchan());
#else
    if (Operation::verbose)
      cerr << "DedispersionPipe::construct_inverse_filterbank set_pfb_all_chan hardcoded true" << endl;
    inverse_filterbank->set_pfb_all_chan(true);
#endif

    if (Operation::verbose)
      cerr << "DedispersionPipe::construct_inverse_filterbank call InverseFilterbankResponse::set_fir_filter" << endl;
    inverse_filterbank_response->set_fir_filter(first_filter);

    if (Operation::verbose)
      cerr << "DedispersionPipe::construct_inverse_filterbank call InverseFilterbankResponse::set_apply_deripple " << config->do_deripple << endl;
    inverse_filterbank_response->set_apply_deripple(config->do_deripple);
  }

  string app = config->report_as;

  if (convolve_when == Filterbank::Config::During)
  {
    if (frequency_response)
    {
      if (Operation::verbose)
        std::cerr << app << ": adding InverseFilterbankResponse to frequency response" << std::endl;
      append_response(frequency_response, inverse_filterbank_response);
      inverse_filterbank->set_response(frequency_response);
    }
  }
  else
  {
    auto freq_res = config->inverse_filterbank.get_freq_res();

    if (Operation::verbose)
      std::cerr << app << ": setting InverseFilterbankResponse frequency resolution to " << freq_res << std::endl;    
    inverse_filterbank->set_response(inverse_filterbank_response);
    inverse_filterbank->set_input_fft_length(freq_res, info);
  }

  pipeline->append(inverse_filterbank.get());
  return output.release();
}


dsp::TimeSeries* dsp::DedispersionPipe::construct_filterbank (TimeSeries* input)
{
  Filterbank::Config::When convolve_when = config->filterbank.get_convolve_when();

  // new storage for filterbank output (must be out-of-place)
  Reference::To<TimeSeries> output = pipeline->new_time_series();

#if HAVE_CUDA
  if (pipeline->run_on_gpu())
    output->set_memory(pipeline->get_memory());
#endif

  config->filterbank.set_device(pipeline->get_memory());
  config->filterbank.set_stream(pipeline->get_gpu_stream());

  // software filterbank constructor
  if (!filterbank)
    filterbank = config->filterbank.create();

  if (!pipeline->get_configuration()->input_buffering)
    filterbank->set_buffering_policy(NULL);

  filterbank->set_input(input);
  filterbank->set_output(output);

  if (convolve_when == Filterbank::Config::During)
  {
    filterbank->set_response(frequency_response);

    // accumulate the passband only if not generating single pulses
    if (config->integrate_passband)
      filterbank->get_engine()->set_passband(passband);
  }

  if (temporal_apodization)
    filterbank->set_temporal_apodization(temporal_apodization);

  if (spectral_apodization)
    filterbank->set_spectral_apodization(spectral_apodization);

  pipeline->append(filterbank.get());
  return output.release();
}

dsp::TimeSeries* dsp::DedispersionPipe::construct_convolution (TimeSeries* input)
{
  if (!convolution)
    convolution = new Convolution;

  if (!pipeline->get_configuration()->input_buffering)
    convolution->set_buffering_policy(NULL);

  convolution->set_device(pipeline->get_memory());
  convolution->set_response(frequency_response);

  if (config->integrate_passband)
    convolution->set_passband(passband);

  if (temporal_apodization)
    convolution->set_temporal_apodization(temporal_apodization);

  if (spectral_apodization)
    convolution->set_spectral_apodization(spectral_apodization);

  Reference::To<TimeSeries> output = pipeline->new_time_series();

  convolution->set_input(input);
  convolution->set_output(output);

#if HAVE_CUDA
  if (pipeline->run_on_gpu())
  {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(pipeline->get_gpu_stream());

    output->set_memory(pipeline->get_memory());
    auto info = pipeline->get_source()->get_info();

    unsigned nchan = info->get_nchan() * config->get_filterbank_nchan();
    if (nchan >= 16)
      convolution->set_engine(new CUDA::ConvolutionEngineSpectral(stream));
    else
      convolution->set_engine(new CUDA::ConvolutionEngine(stream));
  }
#endif

  pipeline->append(convolution.get());

  return output.release();
}


dsp::TimeSeries* dsp::DedispersionPipe::construct_interchan(TimeSeries *input)
{
  if (!config->interchan_dedispersion)
    return input;

  if (Operation::verbose)
    cerr << "DedispersionPipe::construct_interchan correct inter-channel dispersion delay" << endl;

  if (!sample_delay)
    sample_delay = new SampleDelay;

  TimeSeries *output_timeseries = input; // operate in place

  if (pipeline->get_configuration()->get_total_nthread() > 1)
    output_timeseries = pipeline->new_time_series();

  sample_delay->set_input(input);
  sample_delay->set_output(output_timeseries);

  auto delay_function = new Dedispersion::SampleDelay;
  sample_delay->set_function(delay_function);
  if (dedisp)
    dedisp->set_sample_delay(delay_function);

#if HAVE_CUDA
  if (pipeline->run_on_gpu())
  {
    sample_delay->set_engine(
        new CUDA::SampleDelayEngine((cudaStream_t)pipeline->get_gpu_stream()));
    // Note, this assumes the data TimeSeries memory has already been
    // properly set up to use the GPU.
  }
#endif

  pipeline->append(sample_delay.get());

  return output_timeseries;
}

void dsp::DedispersionPipe::prepare()
{
  double dm = config->dispersion_measure;
  double rm = config->rotation_measure;

  if (Operation::verbose)
    cerr << "dsp::DedispersionPipe::prepare DM=" << dm << " RM=" << rm << endl;

  if (config->coherent_dedispersion)
  {
    if (dm == 0.0)
      throw Error(InvalidState, "DedispersionPipe::prepare",
                  "coherent dedispersion enabled, but DM unknown");

    if (dedisp)
      dedisp->set_dispersion_measure(dm);
  }

  if (config->coherent_derotation)
  {
    if (rm == 0.0)
      throw Error (InvalidState, "DedispersionPipe::prepare",
                   "coherent derotation enabled, but RM unknown");

    if (derotate)
      derotate->set_rotation_measure (rm);
  }

  Observation *info = pipeline->get_source()->get_info();
  if (Operation::verbose)
    cerr << "DedispersionPipe::prepare setting Source::info=" << info << " DM= " << dm << endl;
  info->set_dispersion_measure(dm);  
  
  if (Operation::verbose)
    cerr << "DedispersionPipe::prepare setting Source::info=" << info << " RM= " << rm << endl;
  info->set_rotation_measure( rm );
}
