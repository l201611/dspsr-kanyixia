/**************************************************************************2
 *
 *   Copyright (C) 2007 - 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToFold1.h"
#include "dsp/LoadToFoldConfig.h"

#include "dsp/SignalPath.h"
#include "dsp/Scratch.h"
#include "dsp/File.h"

#include "dsp/IOManager.h"

#include "dsp/ExcisionUnpacker.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/WeightedTimeSeries.h"

#include "dsp/ResponseProduct.h"
#include "dsp/Derotation.h"
#include "dsp/RFIFilter.h"
#include "dsp/PolnCalibration.h"

#include "dsp/FIRFilter.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankResponse.h"
#include "dsp/Filterbank.h"
#include "dsp/FilterbankEngineCPU.h"
#include "dsp/SpectralKurtosis.h"
#include "dsp/OptimalFFT.h"
#include "dsp/Resize.h"

#if HAVE_CUDA
#include "dsp/TransferCUDA.h"
#include "dsp/TimeSeriesCUDA.h"
#include "dsp/TransferBitSeriesCUDA.h"
#include "dsp/DetectionCUDA.h"
#include "dsp/FoldCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "dsp/SpectralKurtosisCUDA.h"
#include "dsp/CyclicFoldEngineCUDA.h"
#endif

#include "dsp/PhaseLockedFilterbank.h"
#include "dsp/Detection.h"
#include "dsp/FourthMoment.h"
#include "dsp/Stats.h"

#include "dsp/FoldManager.h"
#include "dsp/Fold.h"
#include "dsp/Subint.h"
#include "dsp/PhaseSeries.h"
#include "dsp/OperationThread.h"

#include "dsp/CyclicFold.h"

#include "dsp/Archiver.h"
#include "dsp/ObservationChange.h"

#include "Pulsar/Archive.h"
#include "Pulsar/TextParameters.h"
#include "Pulsar/SimplePredictor.h"

#include "Error.h"
#include "debug.h"

#include <assert.h>

using namespace std;

static void *const undefined_stream = (void *)-1;

dsp::LoadToFold::LoadToFold(Config *configuration)
try
{
  manage_archiver = true;
  fold_prepared = false;

  if (configuration)
    set_configuration(configuration);
}
catch (Error &error)
{
  throw error += "LoadToFold ctor";
}

dsp::LoadToFold::~LoadToFold()
{
}

void dsp::LoadToFold::set_configuration(Config *configuration)
try
{
  SingleThread::set_configuration(configuration);
  config = configuration;
}
catch (Error &error)
{
  throw error += "dsp::LoadToFold::set_configuration";
}

void dsp::LoadToFold::construct() try
{
  SingleThread::construct();

  if (Operation::verbose)
    cerr << "dsp::LoadToFold::construct" << endl;
  
  DedispersionPipe::config = &(config->dedisp);
  DedispersionPipe::pipeline = this;

  if (source->get_info()->get_detected())
  {
    if (config->dedisp.coherent_derotation)
      throw Error (InvalidParam, "dsp::LoadToFold::construct",
                   "cannot perform phase-coherent Faraday rotation correction\n\t"
                   "because input signal is detected");

    // detected data is handled much more efficiently in TFP order
    if (config->optimal_order && source->get_order_supported(TimeSeries::OrderTFP))
    {
      source->set_output_order(TimeSeries::OrderTFP);
    }

    config->dedisp.coherent_dedispersion = false;

    if (Operation::verbose)
      cerr << "dsp::LoadToFold::construct detected construct_spectral_kurtosis" << endl;

    TimeSeries *data_out = construct_spectral_kurtosis(source_output);

    if (Operation::verbose)
      cerr << "dsp::LoadToFold::construct detected construct_interchan" << endl;

    data_out = construct_interchan(data_out);

    if (Operation::verbose)
      cerr << "dsp::LoadToFold::construct detected build_fold" << endl;
    
    build_fold(data_out);
    return;
  }

  // record the number of operations in signal path
  unsigned noperations = operations.size();

  if (report_vitals())
  {
    config->dedisp.report_as = config->application_name;
  }

  if (source->get_info()->get_type() != Signal::Pulsar)
  {
    // the frequency_response gets messed up by DM=0 sources, like PolnCal
    if (report_vitals())
      cerr << app() << ": disabling coherent dedispersion of non-pulsar signal" << endl;
    config->dedisp.coherent_dedispersion = false;

    if (config->dedisp.coherent_derotation)
      throw Error (InvalidParam, "dsp::LoadToFold::construct",
                   "cannot perform phase-coherent Faraday rotation correction\n\t"
                   "because input signal is not of type 'Pulsar'");
  }

  // accumulate the passband only if not generating single pulses
  config->dedisp.integrate_passband = !config->integration_turns;

  Reference::To<TimeSeries> convolved = DedispersionPipe::construct(source_output);

  if (config->plfb_nbin)
  {
    construct_phased_filterbank(convolved);

    // the phase-locked filterbank does its own detection and folding
    return;
  }

  Reference::To<Fold> presk_fold;
  Reference::To<Archiver> presk_unload;

  TimeSeries *cleaned = construct_spectral_kurtosis(convolved);

  if (config->npol == 0)
    cerr << "dsp::LoadToFold::construct folding voltages" << endl;

  // Cyclic spectrum also detects and folds
  if (config->cyclic_nchan || config->npol == 0)
  {
    build_fold(cleaned);
    return;
  }

  if (!detect)
    detect = new Detection;

  TimeSeries *detected = cleaned;
  detect->set_input(cleaned);
  detect->set_output(cleaned);

  configure_detection(detect, noperations);

  operations.push_back(detect.get());

  if (config->npol == 3 || config->npol == 1 || source->get_info()->get_npol() == 1)
  {
    detected = new_time_series();
    detect->set_output(detected);
  }
  else if (config->fourth_moment)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::construct fourth order moments" << endl;

    FourthMoment *fourth = new FourthMoment;
    operations.push_back(fourth);

    fourth->set_input(detected);
    detected = new_time_series();
    fourth->set_output(detected);
  }

#if HAVE_CUDA
  if (run_on_gpu())
    detected->set_memory(device_memory);
#endif

  build_fold(detected);

  if (presk_fold)
  {
    // presk fold and unload are pushed back after the primary ones are built
    fold.push_back(presk_fold);
    unloader.push_back(presk_unload.get());
  }

  if (config->sk_fold)
  {
    PhaseSeriesUnloader *unload = get_unloader(config->get_nfold());
    unload->set_extension(".sk");

    Reference::To<Fold> skfold;
    build_fold(skfold, unload);

    skfold->set_input(cleaned);
    skfold->prepare(source->get_info());
    skfold->reset();

    fold.push_back(skfold);
    operations.push_back(skfold.get());
  }
}
catch (Error &error)
{
  throw error += "dsp::LoadToFold::construct";
}


dsp::TimeSeries* dsp::LoadToFold::construct_spectral_kurtosis(TimeSeries *input_timeseries)
{
  if (!config->sk_zap)
    return input_timeseries;

  auto convolve_when = config->dedisp.get_convolve_when();

  if (inverse_filterbank && convolve_when == Filterbank::Config::During)
  {
    std::cerr << "dspsr: using InverseFilterbankResponse as zero DM response" << std::endl;
    zero_DM_time_series = new_time_series();
#if HAVE_CUDA
    if (run_on_gpu())
    {
      zero_DM_time_series->set_memory(device_memory);
    }
#endif
    inverse_filterbank->set_zero_DM(true);
    inverse_filterbank->set_zero_DM_output(zero_DM_time_series);
    // the following will be overwritten in InverseFilterbank::prepare
    inverse_filterbank_response->resize(1, 1, config->dedisp.inverse_filterbank.get_freq_res(), 2);
    Reference::To<InverseFilterbankResponse>
        zero_DM_response = new dsp::InverseFilterbankResponse(*inverse_filterbank_response);
    inverse_filterbank->set_zero_DM_response(zero_DM_response);
  }

  if (filterbank && convolve_when == Filterbank::Config::During)
  {
    cerr << "dspsr: using Filterbank with zero DM output timeseries" << endl;
    zero_DM_time_series = new_time_series();
#if HAVE_CUDA
    if (run_on_gpu())
    {
      zero_DM_time_series->set_memory(device_memory);
    }
#endif
    filterbank->set_zero_DM(true);
    filterbank->set_zero_DM_output(zero_DM_time_series);
  }

  if (convolution)
  {
    cerr << "Using zero DM time_series with convolution" << endl;
    zero_DM_time_series = new_time_series();
#if HAVE_CUDA
    if (run_on_gpu())
    {
      zero_DM_time_series->set_memory(device_memory);
    }
#endif
    convolution->set_zero_DM(true);
    convolution->set_zero_DM_output(zero_DM_time_series);
  }

  TimeSeries *cleaned = new_time_series();

  if (!skestimator)
  {
    skestimator = new SpectralKurtosis();
  }

  if (!config->input_buffering)
  {
    skestimator->set_buffering_policy(NULL);
    skestimator->set_zero_DM_buffering_policy(NULL);
  }

  if (config->dedisp.coherent_dedispersion && config->sk_zap)
  {
    skestimator->set_zero_DM(true);
    skestimator->set_zero_DM_input(zero_DM_time_series);
  }

  skestimator->set_input(input_timeseries);
  skestimator->set_output(cleaned);

#if HAVE_CUDA
  if (run_on_gpu())
  {
    // for input buffering
    input_timeseries->set_engine(new CUDA::TimeSeriesEngine(device_memory));
    cleaned->set_memory(device_memory);
    skestimator->set_engine(new CUDA::SpectralKurtosisEngine(device_memory));
  }
#endif

  skestimator->set_detect_time_freq(config->sk_time_freq);
  skestimator->set_detect_freq(config->sk_freq);
  skestimator->set_detect_time(config->sk_time);

  if (config->sk_config != "")
    skestimator->load_configuration(config->sk_config);

  // any command-line options can over-ride/supplement information in config file
  if (config->sk_m.size())
    skestimator->set_M(config->sk_m);

  if (config->sk_noverlap.size())
    skestimator->set_noverlap(config->sk_noverlap);

  if (config->sk_std_devs.size())
    skestimator->set_thresholds(config->sk_std_devs);

  // if either chan_start or chan_end is specified (non-zero)
  if (config->sk_chan_start + config->sk_chan_end != 0)
    skestimator->set_channel_range(config->sk_chan_start, config->sk_chan_end);

  skestimator->set_omit_outliers(config->sk_omit_outliers);

  operations.push_back(skestimator.get());

  return cleaned;
}

void dsp::LoadToFold::construct_phased_filterbank(TimeSeries *input)
{
  // Set up output
  Archiver *archiver = new Archiver;
  unloader.resize(1);
  unloader[0] = archiver;
  prepare_archiver(archiver);

  if (!phased_filterbank)
  {
    if (output_subints())
    {
      Subint<PhaseLockedFilterbank> *sub_plfb = new Subint<PhaseLockedFilterbank>;

      if (config->integration_length)
      {
        sub_plfb->set_subint_seconds(config->integration_length);
      }

      else if (config->integration_turns)
      {
        sub_plfb->set_subint_turns(config->integration_turns);
        sub_plfb->set_fractional_pulses(config->fractional_pulses);
      }

      sub_plfb->set_unloader(unloader[0]);

      phased_filterbank = sub_plfb;
    }
    else
    {
      phased_filterbank = new PhaseLockedFilterbank;
    }
  }

  phased_filterbank->set_nbin(config->plfb_nbin);
  phased_filterbank->set_npol(config->npol);

  if (config->plfb_nchan)
    phased_filterbank->set_nchan(config->plfb_nchan);

  phased_filterbank->set_input(input);

  if (!phased_filterbank->has_output())
    phased_filterbank->set_output(new PhaseSeries);

  phased_filterbank->bin_divider.set_reference_phase(config->reference_phase);

  // Make dummy fold instance so that polycos get created
  fold.resize(1);
  fold[0] = new dsp::Fold;
  if (config->folding_period)
    fold[0]->set_folding_period(config->folding_period);
  if (config->ephemerides.size() > 0)
    fold[0]->set_pulsar_ephemeris(config->ephemerides[0]);
  else if (config->predictors.size() > 0)
    fold[0]->set_folding_predictor(config->predictors[0]);
  fold[0]->set_output(phased_filterbank->get_output());
  fold[0]->prepare(source->get_info());

  operations.push_back(phased_filterbank.get());

  path.resize(1);
  path[0] = new SignalPath(operations);
}

double get_dispersion_measure(const Pulsar::Parameters *parameters)
{
  const Pulsar::TextParameters *teph;
  teph = dynamic_cast<const Pulsar::TextParameters *>(parameters);
  if (teph)
  {
    double dm = 0.0;
    teph->get_value(dm, "DM");
    return dm;
  }

  throw Error(InvalidState, "get_dispersion_measure (Pulsar::Parameters*)",
              "unknown Parameters class");
}

double get_rotation_measure (const Pulsar::Parameters* parameters)
{
  const Pulsar::TextParameters* teph;
  teph = dynamic_cast<const Pulsar::TextParameters*>(parameters);
  if (teph)
  {
    double rm = 0.0;
    teph->get_value (rm, "RM");
    return rm;
  }

  throw Error (InvalidState, "get_rotation_measure (Pulsar::Parameters*)",
               "unknown Parameters class");
}

void dsp::LoadToFold::prepare_fold ()
{
  if (fold_prepared)
    return;

  for (unsigned ifold = 0; ifold < fold.size(); ifold++)
    fold[ifold]->prepare(source->get_info());

  fold_prepared = true;
}

MJD dsp::LoadToFold::parse_epoch(const std::string &epoch_string)
{
  MJD epoch;

  if (epoch_string == "start")
  {
    epoch = source->get_info()->get_start_time();
    epoch += source->get_current_time();

    if (Operation::verbose)
      cerr << "dsp::LoadToFold::parse reference epoch=start_time="
           << epoch.printdays(13) << endl;
  }
  else if (!epoch_string.empty())
  {
    epoch = MJD(epoch_string);
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::parse reference epoch="
           << epoch.printdays(13) << endl;
  }

  return epoch;
}

void dsp::LoadToFold::prepare()
{
  assert(fold.size() > 0);

  prepare_fold();

  if (log)
  {
    for (unsigned iul = 0; iul < unloader.size(); iul++)
      unloader[iul]->set_cerr(*log);
  }

  const Pulsar::Predictor *predictor = 0;
  if (fold[0]->has_folding_predictor())
    predictor = fold[0]->get_folding_predictor();

  if (phased_filterbank)
    phased_filterbank->bin_divider.set_predictor(predictor);

  const Pulsar::Parameters *parameters = 0;
  if (fold[0]->has_pulsar_ephemeris())
    parameters = fold[0]->get_pulsar_ephemeris();

  double dm = 0.0;

  if (config->dedisp.dispersion_measure != -1.0)
  {
    dm = config->dedisp.dispersion_measure;
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::prepare config DM=" << dm << endl;
  }
  else if (parameters)
  {
    dm = get_dispersion_measure(parameters);
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::prepare ephem DM=" << dm << endl;
  }

  double rm = 0.0;

  if (config->dedisp.rotation_measure)
  {
    rm = config->dedisp.rotation_measure;
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::prepare config RM=" << rm << endl;
  }
  else if (parameters) try
  {
    rm = get_rotation_measure (parameters);
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::prepare ephem RM=" << rm << endl;
  }
  catch (Error& error)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::prepare could not parse RM from ephemeris (ignored)" << endl;
  }

  // --repeat must reset the dm and rm when the input is re-opened
  config->dedisp.dispersion_measure = dm;
  config->dedisp.rotation_measure = rm;

  DedispersionPipe::prepare();

  if (log)
  {
    if (frequency_response)
      frequency_response->set_cerr(*log);
  }

  MJD fold_reference_epoch = parse_epoch(config->reference_epoch);

  for (unsigned ifold = 0; ifold < fold.size(); ifold++)
  {
    if (ifold < path.size())
    {
      Reference::To<Extensions> extensions = new Extensions;
      extensions->add_extension(path[ifold]);

      for (unsigned iop = 0; iop < operations.size(); iop++)
        operations[iop]->add_extensions(extensions);

      fold[ifold]->get_output()->set_extensions(extensions);
    }

    fold[ifold]->set_reference_epoch(fold_reference_epoch);
  }

  SingleThread::prepare();
}

//! Set the block size during prepare method
void dsp::LoadToFold::prepare_block_size()
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold::prepare_block_size" << endl;
  config->dedisp.input_buffering = config->input_buffering; // kludge
  DedispersionPipe::prepare(this);
}

void dsp::LoadToFold::end_of_data()
{
  // ensure that remaining threads are not left waiting
  for (unsigned ifold = 0; ifold < fold.size(); ifold++)
    fold[ifold]->finish();

  SingleThread::end_of_data();
}

void setup(dsp::Fold *fold)
{
  if (!fold->has_output())
    fold->set_output(new dsp::PhaseSeries);
}

template <class T>
T *setup(Reference::To<dsp::Fold> &ptr)
{
  if (!ptr)
    ptr = new T;

  // ensure that the current folder is of type T
  T *derived = dynamic_cast<T *>(ptr.ptr());

  if (!derived)
    throw Error(InvalidState, "setup", "Fold not of expected type");

  setup(derived);

  return derived;
}

const char *multifold_error =
    "Folding more than one pulsar and output archive filename set to\n"
    "\t%s\n"
    "The multiple output archives would over-write each other.\n";

void dsp::LoadToFold::build_fold(TimeSeries *to_fold)
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold::build_fold" << endl;

  if (config->pdmp_output)
  {
    Stats *stats = new Stats;
    stats->set_input(to_fold);
    operations.push_back(stats);
  }

  size_t nfold = config->get_nfold();

  if (nfold > 1 && !config->archive_filename.empty())
    throw Error(InvalidState, "dsp::LoadToFold::build_fold",
                multifold_error, config->archive_filename.c_str());

  if (Operation::verbose)
    cerr << "dsp::LoadToFold::build_fold nfold=" << nfold << endl;

  /*
    To work on solving https://sourceforge.net/p/dspsr/bugs/93/
    uncomment the following line
  */
  // fold_manager = new FoldManager;

  fold.resize(nfold);
  path.resize(nfold);
  unloader.resize(nfold);

  if (config->asynchronous_fold)
    asynch_fold.resize(nfold);

  for (unsigned ifold = 0; ifold < nfold; ifold++)
  {
    build_fold(fold[ifold], get_unloader(ifold));

    /*
      path must be built before fold[ifold] is added to operations vector
      so that each path will contain only one Fold instance.
    */

    path[ifold] = new SignalPath(operations);
    path[ifold]->add(fold[ifold]);

    configure_fold(ifold, to_fold);
  }

  if (fold_manager)
    operations.push_back(fold_manager.get());
  else
  {
    for (unsigned i = 0; i < fold.size(); i++)
      operations.push_back(fold[i].get());
  }
}

dsp::PhaseSeriesUnloader *
dsp::LoadToFold::get_unloader(unsigned ifold)
{
  if (ifold == unloader.size())
    unloader.push_back(NULL);

  if (!unloader.at(ifold))
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::get_unloader prepare new Archiver" << endl;

    Archiver *archiver = new Archiver;
    unloader[ifold] = archiver;
    prepare_archiver(archiver);
  }

  return unloader.at(ifold);
}

void dsp::LoadToFold::build_fold(Reference::To<Fold> &fold, PhaseSeriesUnloader *unloader)
try
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFold::build_fold input ptr=" << fold.ptr() << " unloader ptr=" << unloader << endl;

  if (!output_subints())
  {
    if (config->cyclic_nchan)
    {
      if (Operation::verbose)
        cerr << "dsp::LoadToFold::build_fold prepare CyclicFold" << endl;

      CyclicFold *cs = new CyclicFold;
      cs->set_mover(config->cyclic_mover);
      cs->set_nchan(config->cyclic_nchan);
      cs->set_npol(config->npol);

      fold = cs;
    }
    else
    {
      if (Operation::verbose)
        cerr << "dsp::LoadToFold::build_fold prepare Fold" << endl;

      fold = new Fold;
    }
  }
  else if (config->cyclic_nchan)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::build_fold prepare Subint<CyclicFold>" << endl;

    Subint<CyclicFold> *subfold = new Subint<CyclicFold>;

    subfold->set_mover(config->cyclic_mover);
    subfold->set_nchan(config->cyclic_nchan);
    subfold->set_npol(config->npol);

    if (config->integration_length)
    {
      subfold->set_subint_seconds(config->integration_length);

      if (config->minimum_integration_length > 0)
        unloader->set_minimum_integration_length(config->minimum_integration_length);
    }
    else
      throw Error(InvalidState, "dsp::LoadToFold::build_fold",
                  "Single-pulse cyclic spectra not supported");

    subfold->set_unloader(unloader);

    fold = subfold;
  }
  else
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::build_fold prepare Subint<Fold>" << endl;

    Subint<Fold> *subfold = new Subint<Fold>;

    if (config->integration_length)
    {
      subfold->set_subint_seconds(config->integration_length);

      if (config->minimum_integration_length > 0)
        unloader->set_minimum_integration_length(config->minimum_integration_length);

      MJD reference_epoch = parse_epoch(config->integration_reference_epoch);
      subfold->set_subint_reference_epoch(reference_epoch);
    }
    else
    {
      subfold->set_subint_turns(config->integration_turns);
      subfold->set_fractional_pulses(config->fractional_pulses);
    }

    subfold->set_unloader(unloader);

    fold = subfold;
  }

  setup(fold);

  if (Operation::verbose)
    cerr << "dsp::LoadToFold::build_fold configuring ptr=" << fold.ptr() << endl;

  if (config->nbin)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::build_fold nbin=" << config->nbin << endl;
    fold->set_nbin(config->nbin);
    fold->set_force_sensible_nbin(config->force_sensible_nbin);
  }

  if (config->reference_phase)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::build_fold reference_phase=" << config->reference_phase << endl;
    fold->set_reference_phase(config->reference_phase);
  }

  if (config->folding_period)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::build_fold folding_period=" << config->folding_period << endl;
    fold->set_folding_period(config->folding_period);
  }

  if (Operation::verbose)
    cerr << "dsp::LoadToFold::build_fold output ptr=" << fold.ptr() << endl;
}
catch (Error &error)
{
  throw error += "dsp::LoadToFold::build_fold";
}

void dsp::LoadToFold::configure_detection(Detection *detect, unsigned noperations)
{
#if HAVE_CUDA
  //bool run_on_gpu = thread_id < config->get_cuda_ndevice();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(gpu_stream);

  if (run_on_gpu())
  {
    config->ndim = 2;
    detect->set_engine(new CUDA::DetectionEngine(stream));
  }
#endif

  if (source->get_info()->get_npol() == 1)
  {
    cerr << "Only single polarization detection available" << endl;
    detect->set_output_state(Signal::PP_State);
  }
  else
  {
    if (config->fourth_moment)
    {
      detect->set_output_state(Signal::Stokes);
      detect->set_output_ndim(4);
    }
    else if (config->npol == 4)
    {
      detect->set_output_state(Signal::Coherence);
      detect->set_output_ndim(config->ndim);
    }
    else if (config->npol == 3)
      detect->set_output_state(Signal::NthPower);
    else if (config->npol == 2)
      detect->set_output_state(Signal::PPQQ);
    else if (config->npol == 1)
      detect->set_output_state(Signal::Intensity);
    else
      throw Error(InvalidState, "dsp::LoadToFold::construct",
                  "invalid npol config=%d input=%d",
                  config->npol, source->get_info()->get_npol());
  }

  if (detect->get_order_supported(TimeSeries::OrderTFP) && noperations == operations.size()) // no operations yet added
  {
    if (source->get_order_supported(TimeSeries::OrderTFP))
    {
      cerr << "unpack more efficiently in TFP order" << endl;
      source->set_output_order(TimeSeries::OrderTFP);
    }
  }
}

void dsp::LoadToFold::configure_fold(unsigned ifold, TimeSeries *to_fold)
{
  Reference::To<ObservationChange> change;

  if (ifold && ifold <= config->additional_pulsars.size())
  {
    change = new ObservationChange;
    change->set_source(config->additional_pulsars[ifold - 1]);
  }

  if (ifold < config->ephemerides.size())
  {
    if (!change)
      change = new ObservationChange;

    Pulsar::Parameters *ephem = config->ephemerides[ifold];
    change->set_source(ephem->get_name());
    change->set_dispersion_measure(get_dispersion_measure(ephem));

    fold[ifold]->set_pulsar_ephemeris(config->ephemerides[ifold]);
  }

  if (ifold < config->predictors.size())
  {
    fold[ifold]->set_folding_predictor(config->predictors[ifold]);

    Pulsar::SimplePredictor *simple = dynamic_kast<Pulsar::SimplePredictor>(config->predictors[ifold]);

    if (simple)
    {
      config->dedisp.dispersion_measure = simple->get_dispersion_measure();

      if (simple->get_reference_epoch() == MJD::zero)
      {
        // ensure that all threads use the same reference epoch

        MJD reference_epoch = source->get_info()->get_start_time();
        reference_epoch += source->get_current_time();

        simple->set_reference_epoch(reference_epoch);
      }

      if (!change)
        change = new ObservationChange;

      change->set_source(simple->get_name());
      change->set_dispersion_measure(simple->get_dispersion_measure());
    }
  }

  fold[ifold]->set_input(to_fold);

  if (change)
    fold[ifold]->set_change(change);

  if (ifold && ifold <= config->additional_pulsars.size())
  {
    if (!change)
      change = new ObservationChange;

    /*
      If additional pulsar names have been specified, then Fold::prepare
      will have retrieved an ephemeris, and the DM from this ephemeris
      should make its way into the folded profile.
    */
    const Pulsar::Parameters *ephem = fold[ifold]->get_pulsar_ephemeris();
    change->set_dispersion_measure(get_dispersion_measure(ephem));
  }

  // fold[ifold]->reset();

  if (config->asynchronous_fold)
    asynch_fold[ifold] = new OperationThread(fold[ifold].get());
  else if (fold_manager)
    fold_manager->manage(fold[ifold]);

#if HAVE_CUDA
  if (gpu_stream != undefined_stream)
  {
    cudaStream_t stream = (cudaStream_t)gpu_stream;
    if (config->cyclic_nchan)
      fold[ifold]->set_engine(new CUDA::CyclicFoldEngineCUDA(stream));
    else
      fold[ifold]->set_engine(new CUDA::FoldEngine(stream, config->sk_zap));
  }
#endif
}

void dsp::LoadToFold::prepare_archiver(Archiver *archiver)
{
  bool multiple_outputs = output_subints() && ((config->subints_per_archive > 0) || (config->single_archive == false));

  archiver->set_command_line(config->command_line);

  archiver->set_archive_class(config->archive_class.c_str());
  if (config->archive_class_specified_by_user)
    archiver->set_force_archive_class(true);

  if (output_subints() && config->single_archive)
  {
    cerr << "dspsr: Single archive with multiple sub-integrations" << endl;
    archiver->set_use_single_archive(true);
  }

  if (output_subints() && config->subints_per_archive)
  {
    cerr << "dspsr: Archives with " << config->subints_per_archive << " sub-integrations" << endl;
    archiver->set_use_single_archive(true);
    archiver->set_subints_per_file(config->subints_per_archive);
  }

  if (config->integration_turns || !config->dynamic_extensions)
    archiver->set_store_dynamic_extensions(false);

  FilenameEpoch *epoch_convention = 0;
  FilenameSequential *index_convention = 0;

  if (config->filename_convention == "mjd")
    archiver->set_convention(new FilenameMJD(13));
  else if (config->concurrent_archives())
    archiver->set_convention(new FilenamePulse);
  else
  {
    // If there is only one output file, use epoch convention.
    if (!multiple_outputs)
      archiver->set_convention(epoch_convention = new FilenameEpoch);

    // If archive_filename was specified, figure out whether
    // it represents a date string or not by looking for '%'
    // characters.
    else if (!config->archive_filename.empty())
    {
      if (config->archive_filename.find('%') == string::npos)
        archiver->set_convention(index_convention = new FilenameSequential);
      else
        archiver->set_convention(epoch_convention = new FilenameEpoch);
    }

    // Default to epoch convention otherwise.
    else
      archiver->set_convention(epoch_convention = new FilenameEpoch);
  }

  if (epoch_convention && output_subints() && (config->single_archive || config->subints_per_archive))
    epoch_convention->report_unload = false;

  unsigned integer_seconds = unsigned(config->integration_length);

  if (config->integration_length && config->integration_turns)
    throw Error(InvalidState, "dsp::LoadToFold::prepare_archiver",
                "cannot set integration length in single pulse mode");

  if (config->integration_length &&
      config->integration_length == integer_seconds &&
      epoch_convention)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFold::prepare_archiver integer_seconds="
           << integer_seconds << " in output filenames" << endl;

    epoch_convention->set_integer_seconds(integer_seconds);
  }

  if (!config->archive_filename.empty())
  {
    if (epoch_convention)
      epoch_convention->set_datestr_pattern(config->archive_filename);
    else if (index_convention)
      index_convention->set_base_filename(config->archive_filename);
    else
      throw Error(InvalidState, "dsp::LoadToFold::prepare_archiver",
                  "cannot set archive filename in single pulse mode");
  }

  archiver->set_archive_software("dspsr");

  if (sample_delay)
    archiver->set_archive_dedispersed(true);

  if (config->jobs.size())
    archiver->set_script(config->jobs);

  if (fold.size() > 1)
    archiver->set_path_add_source(true);

  if (!config->archive_extension.empty())
    archiver->set_extension(config->archive_extension);

  auto input_source = dynamic_cast<InputSource<Input> *>(source.get());
  if (input_source)
    archiver->set_prefix(input_source->get_input()->get_prefix());
}

bool dsp::LoadToFold::output_subints() const
{
  return config && (config->integration_turns || config->integration_length);
}

void dsp::LoadToFold::share(SingleThread *other)
{
  SingleThread::share(other);

  if (Operation::verbose)
    cerr << "dsp::LoadToFold::share other=" << other << endl;

  LoadToFold *thread = dynamic_cast<LoadToFold *>(other);

  if (!thread)
    throw Error(InvalidParam, "dsp::LoadToFold::share", "other thread is not a LoadToFold instance");

  unsigned nfold = thread->fold.size();

  assert(nfold == fold.size());

  for (unsigned ifold = 0; ifold < nfold; ifold++)
  {
    Fold *from = thread->fold[ifold];
    Fold *to = fold[ifold];

    // ... simply share the ephemeris
    if (from->has_pulsar_ephemeris())
      to->set_pulsar_ephemeris(from->get_pulsar_ephemeris());

    // ... but each fold should have its own predictor
    if (from->has_folding_predictor())
      to->set_folding_predictor(from->get_folding_predictor()->clone());
  }

  //
  // only the first thread must manage archival
  //
  if (output_subints())
    manage_archiver = false;
}

void dsp::LoadToFold::finish()
try
{
  if (phased_filterbank)
  {
    cerr << "Calling PhaseLockedFilterbank::normalize_output" << endl;
    phased_filterbank->normalize_output();
  }

  if (!output_subints())
  {
    if (!unloader.size())
      throw Error(InvalidState, "dsp::LoadToFold::finish", "no unloader");

    for (unsigned i = 0; i < fold.size(); i++)
    {
      Archiver *archiver = dynamic_cast<Archiver *>(unloader[0].get());
      if (!archiver)
        throw Error(InvalidState, "dsp::LoadToFold::finish",
                    "unloader is not an archiver (single integration)");

      /*
        In multi-threaded applications, the thread that calls the
        finish method may not be the thread that called the prepare
        method.
      */

      if (Operation::verbose)
        cerr << "Creating archive " << i + 1 << endl;

      if (phased_filterbank)
        archiver->unload(phased_filterbank->get_output());
      else
        archiver->unload(fold[i]->get_result());
    }
  }

  SingleThread::finish();

}
catch (Error &error)
{
  throw error += "dsp::LoadToFold::finish";
}
