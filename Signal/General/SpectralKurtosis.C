/***************************************************************************
 *
 *   Copyright (C) 2010 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/SpectralKurtosis.h"
#include "dsp/InputBuffering.h"
#include "dsp/SKLimits.h"

#if HAVE_YAMLCPP
#include <yaml-cpp/yaml.h>
#endif

#include <errno.h>
#include <assert.h>
#include <string.h>

#include <fstream>
#include <algorithm>

using namespace std;

dsp::SpectralKurtosis::SpectralKurtosis()
 : Transformation<TimeSeries,TimeSeries>("SpectralKurtosis", outofplace)
{
  resolution.resize(1);
  resolution[0].set_M(128);
  resolution[0].set_noverlap(1);
  resolution[0].set_std_devs(3.0);

  debugd = 1;

  sums = new WeightedTimeSeries;
  zapmask = new BitSeries;

  set_buffering_policy(new InputBuffering(this));
  set_zero_DM_buffering_policy(new InputBuffering(&zero_DM_input_container));
}

dsp::SpectralKurtosis::~SpectralKurtosis ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::SpectralKurtosis~" << endl;

  cerr << "SpectralKurtosis Zapped: tscr=" << SK_freq.fraction_zapped() * 100.0 << "\%" << endl;
  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    cerr << "\t" << ires << ": M=" << resolution[ires].get_M()
         << " skfb=" << resolution[ires].get_count_time_freq().fraction_zapped() * 100.0 << "\%"
         << " fscr=" << resolution[ires].get_count_time().fraction_zapped() * 100.0 << "\%" << endl;
  }

  delete sums;
  delete zapmask;
}

bool dsp::SpectralKurtosis::get_order_supported (TimeSeries::Order order) const
{
  if (order == TimeSeries::OrderFPT || order == TimeSeries::OrderTFP)
    return true;
  return false;
}


void dsp::SpectralKurtosis::set_engine (Engine* _engine)
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::set_engine()" << endl;
  engine = _engine;
}

#if HAVE_YAMLCPP
void parse_ranges (YAML::Node node,
		   vector< pair<unsigned,unsigned> >& ranges)
{
  if (!node.IsSequence())
    throw Error (InvalidState, "parse_ranges",
		 "YAML::Node is not a sequence");

  if (node[0].IsSequence())
  {
    for (unsigned i=0; i < node.size(); i++)
      parse_ranges (node[i], ranges);
    return;
  }

  std::pair<unsigned,unsigned> range;
  range = node.as< std::pair<unsigned,unsigned> >();
  ranges.push_back(range);
}
#endif

//! Load configuration from YAML filename
void dsp::SpectralKurtosis::load_configuration (const std::string& filename)
{
#if HAVE_YAMLCPP
  YAML::Node node = YAML::LoadFile(filename);

  unsigned nres = 1;

  // different resolutions are specified in a sequence
  if (node.IsSequence())
    nres = node.size();

#if _DEBUG
  cerr << "dsp::SpectralKurtosis::load_configuration " << filename << endl;
  cerr << "dsp::SpectralKurtosis::load_configuration nodes=" << nres << endl;
#endif

  resolution.resize( nres );

  for (unsigned ires=0; ires<nres; ires++)
  {
    YAML::Node one;
    if (node.IsSequence())
      one = node[ires];
    else
      one = node;

    if ( !one["M"] )
      throw Error (InvalidState, "SpectralKurtosis::load_configuration",
		   "M not specified for resolution[%u]", ires);

    resolution[ires].set_M( one["M"].as<unsigned>() );

    // the rest are optional

    if ( one["overlap"] )
      resolution[ires].set_noverlap (one["overlap"].as<unsigned>());

    if ( one["exclude"] )
      parse_ranges( one["exclude"], resolution[ires].exclude );

    if ( one["include"] )
      parse_ranges( one["include"], resolution[ires].include );
  }

#else
  throw Error (InvalidState, "dsp::SpectralKurtosis::load_configuration",
               "not implemented - requires yaml-cpp");
#endif
}

void dsp::SpectralKurtosis::set_zero_DM_input (TimeSeries* _zero_DM_input)
{
  zero_DM_input_container.set_input(_zero_DM_input);
}

bool dsp::SpectralKurtosis::has_zero_DM_input () const {
  return zero_DM_input_container.has_input();
}

const dsp::TimeSeries* dsp::SpectralKurtosis::get_zero_DM_input () const {
  return zero_DM_input_container.get_input();
}

dsp::TimeSeries* dsp::SpectralKurtosis::get_zero_DM_input () {
  return const_cast<dsp::TimeSeries*>(zero_DM_input_container.get_input());
}

void dsp::SpectralKurtosis::set_M (const std::vector<unsigned>& M)
{
  // cerr << "SpectralKurtosis::set_M size=" << M.size() << endl;

  resize_resolution (M.size());
  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    unsigned jres = (M.size() == 1) ? 0 : ires;
    resolution[ires].set_M(M[jres]);
  }
}

void dsp::SpectralKurtosis::set_noverlap (const std::vector<unsigned>& noverlap)
{
  // cerr << "SpectralKurtosis::set_noverlap size=" << noverlap.size() << endl;

  resize_resolution (noverlap.size());
  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    unsigned jres = (noverlap.size() == 1) ? 0 : ires;
    resolution[ires].set_noverlap(noverlap[jres]);
  }
}

void dsp::SpectralKurtosis::set_thresholds (const std::vector<float>& std_devs)
{
  resize_resolution (std_devs.size());
  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    unsigned jres = (std_devs.size() == 1) ? 0 : ires;
    resolution[ires].set_std_devs(std_devs[jres]);
  }
}

void dsp::SpectralKurtosis::resize_resolution (unsigned nres)
{
  if (nres == 1)
    return;

  if (resolution.size() == 1)
    resolution.resize (nres);
  else if (nres != resolution.size())
    throw Error (InvalidParam, "dsp::SpectralKurtosis::resize_resolution",
                 "size mismatch n=%u != size=%u", nres, resolution.size());
}

bool dsp::SpectralKurtosis::by_M (const Resolution& A, const Resolution& B)
{
  return A.get_M() < B.get_M();
}

void dsp::SpectralKurtosis::Resolution::set_M (unsigned _M)
{
  thresholds.resize(0);
  M=_M;
}

void dsp::SpectralKurtosis::Resolution::set_Nd (unsigned _Nd)
{
  thresholds.resize(0);
  Nd=_Nd;
}

void dsp::SpectralKurtosis::Resolution::set_std_devs (float _std_devs)
{
  thresholds.resize(0);
  std_devs=_std_devs;
}

void dsp::SpectralKurtosis::Resolution::set_noverlap (unsigned _noverlap)
{
  thresholds.resize(0);
  noverlap = _noverlap;
}

const std::vector<float>&
dsp::SpectralKurtosis::Resolution::get_thresholds () const
{
  if (thresholds.size() == 0)
    set_thresholds ();

  return thresholds;
}

void dsp::SpectralKurtosis::Resolution::prepare (uint64_t ndat)
{
  if (M % noverlap)
    throw Error (InvalidState, "dsp::SpectralKurtosis::Resolution::prepare",
                 "noverlap=%u does not divide M=%u", noverlap, M);

  /* NZAPP-208: WvS - 2020-05-19
    with reference to diagram in SpectralKurtosisInputBuffering.pptx
      idat_off = overlap_offset
      Nblock = npart
  */

  // overlap_offset = idat_off in SpectralKurtosisInputBuffering.pptx
  uint64_t idat_off = overlap_offset = M / noverlap;
  uint64_t idat_start = M - idat_off;

  npart = 0;
  output_ndat = 0;

  if (ndat < idat_start)
    return;

  uint64_t Nblock = (ndat-idat_start) / idat_off;  // number of overlapping blocks
  uint64_t idat_end = Nblock * idat_off;

  if (idat_end < idat_start)
    return;

  npart = Nblock;
  output_ndat = idat_end - idat_start;
}

void dsp::SpectralKurtosis::Resolution::compatible (Resolution& smaller)
{
  if (M % smaller.M)
    throw Error (InvalidState, "dsp::SpectralKurtosis::Resolution::compatible",
                 "this.M=%u not divisible by smaller.M=%u", M, smaller.M);

  if (overlap_offset % smaller.overlap_offset)
    throw Error (InvalidState, "dsp::SpectralKurtosis::Resolution::compatible",
                 "this.overlap_offset=%u not divisible by smaller.overlap_offset=%u",
                 overlap_offset, smaller.overlap_offset);
}


void dsp::SpectralKurtosis::Resolution::add_include
(const std::pair<unsigned, unsigned>& range)
{
  include.push_back(range);
}

void dsp::SpectralKurtosis::Resolution::add_exclude
(const std::pair<unsigned, unsigned>& range)
{
  exclude.push_back(range);
}

void dsp::SpectralKurtosis::Resolution::increment_time_freq (const Count& count)
{
  SK_time_freq += count;
}

auto dsp::SpectralKurtosis::Resolution::get_count_time_freq () const -> const Count&
{
  return SK_time_freq;
}

void dsp::SpectralKurtosis::Resolution::increment_time (const Count& count)
{
  SK_time += count;
}

auto dsp::SpectralKurtosis::Resolution::get_count_time () const -> const Count&
{
  return SK_time;
}

void set_mask (vector<bool>& mask, const vector< pair<unsigned,unsigned> >& ranges, bool value)
{
  for (unsigned irange=0; irange<ranges.size(); irange++)
  {
    unsigned first = ranges[irange].first;
    unsigned last = ranges[irange].second;

    assert (last < mask.size());

    for (unsigned imask=first; imask <= last; imask++)
      mask[imask] = value;
  }
}

//! Get the channels to be zapped
const std::vector<bool>&
dsp::SpectralKurtosis::Resolution::get_channels (unsigned nchan) const
{
  if (nchan == channels.size())
    return channels;

  channels.resize (nchan);

  if (include.size() == 0)
    std::fill (channels.begin(), channels.end(), true);
  else
    std::fill (channels.begin(), channels.end(), false);

  set_mask (channels, include, true);
  set_mask (channels, exclude, false);

  return channels;
}

/*
 * These are preparations that could be performed once at the start of
 * the data processing
 */
void dsp::SpectralKurtosis::prepare ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::prepare()" << endl;

  std::sort (resolution.begin(), resolution.end(), by_M);

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    resolution[ires].prepare();
    if (ires > 0)
      resolution[ires].compatible( resolution[ires-1] );
  }

  nchan = input->get_nchan();
  npol = input->get_npol();
  ndim = input->get_ndim();

  integrated_Nd = 1;   // this is the case for complex-valued input data (state == Analytic_)
  sums_npol = npol;

  if (input->get_state() == Signal::Nyquist)
  {
    throw Error (InvalidState, "dsp::SpectralKurtosis::prepare", "handling real-valued voltage input data not implemented");
    // I think it would require something like passing effective_M = M / 2 to SKLimits
  }
  else if (input->get_detected())
  {
    // determine the number of intensity instances integrated per sample

    double Nyquist_rate = fabs(input->get_bandwidth()) * 1e6 / input->get_nchan();
    double sampling_rate = input->get_rate();
    integrated_Nd = round(Nyquist_rate / sampling_rate);

    // integrating both polarizations doubles the number of intensity instances
    if (input->get_state() == Signal::Stokes || input->get_state() == Signal::Intensity)
    {
      integrated_Nd *= 2;
      sums_npol = 1;
    }
    else if (input->get_state() == Signal::Coherence || input->get_state() == Signal::PPQQ)
    {
      sums_npol = 2;
    }
    else
      throw Error (InvalidState, "dsp::SpectralKurtosis::prepare", "handling of input state=" + tostring(input->get_state()) + " not implemented");

    if (verbose)
      cerr << "dsp::SpectralKurtosis::prepare generalized spectral kurtosis Nd=" << integrated_Nd << endl;
  }

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    resolution[ires].set_Nd (integrated_Nd);
  }

  Memory * memory = const_cast<Memory *>(input->get_memory());
  sums->set_memory (memory);
  zapmask->set_memory (memory);

  // resolution vector sorted by M
  unsigned max_M = resolution.back().get_M();

  if (has_buffering_policy())
  {
    get_buffering_policy()->set_maximum_samples (max_M);
    if (zero_DM)
      get_zero_DM_buffering_policy()->set_maximum_samples (max_M);
  }

  if (engine)
  {
    engine->setup ();
  }

  // ensure output containers are configured correctly
  prepare_output ();

  prepared = true;
}

/*! ensure output parameters are configured correctly */
void dsp::SpectralKurtosis::prepare_output ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::prepare_output()" << endl;

  // Resolution::compatible enforces decreasing overlap_offset order
  unsigned min_offset = resolution.front().get_overlap_offset();
  unsigned max_npart = resolution.front().get_npart();

  double mask_rate = input->get_rate() / min_offset;

  sums->copy_configuration (get_input());
  sums->set_ndat_per_weight(1);
  sums->set_nchan_weight(nchan);
  sums->set_order (TimeSeries::OrderTFP);  // stored in TFP order
  sums->set_scale (1.0);                   // no scaling
  sums->set_rate (mask_rate);              // rate is *= noverlap/M

  if (sums_npol == 2)
    sums->set_state (Signal::PPQQ);
  else
    sums->set_state (Signal::Intensity);

  sums->set_npol (sums_npol);
  sums->set_ndim (2);                      // S1_sum and S2_sum

  double tscrunch_mask_rate = mask_rate;

  if (max_npart > 0)
    tscrunch_mask_rate /= max_npart;

  // zap mask has same configuration as sums with following changes
  zapmask->copy_configuration (sums);
  zapmask->set_nbit (8);
  zapmask->set_npol (1);
  zapmask->set_ndim (1);

  // configure output timeseries (out-of-place) to match input
  output->copy_configuration (get_input());

  /* NZAPP-208 - see SpectralKurtosisInputBuffering.pptx */
  unsigned idat_off = resolution.back().get_overlap_offset();
  unsigned idat_start = resolution.back().get_M() - idat_off;

  output->set_input_sample (input->get_input_sample () + idat_start);
  output->change_start_time (idat_start);
  delay_time = idat_start / output->get_rate();

  if (zero_DM)
  {
    if (get_zero_DM_input()->get_input_sample() != input->get_input_sample ())
      throw Error (InvalidState, "dsp::SpectralKurtosis::prepare_output",
                   "mismatch between normal and zero_DM input samples");
  }
}

double dsp::SpectralKurtosis::get_delay_time () const
{
  return delay_time;
}


/* ensure containers have correct dynamic size */
void dsp::SpectralKurtosis::reserve ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::reserve()" << endl;

  const uint64_t ndat = input->get_ndat();

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    resolution[ires].prepare (ndat);
    if (ires > 0)
      resolution[ires].compatible( resolution[ires-1] );
  }

  unsigned max_npart = resolution.front().get_npart();
  unsigned min_output_ndat = resolution.back().get_output_ndat();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::reserve input_ndat=" << ndat
         << " max npart=" << max_npart
         << " output_ndat=" << min_output_ndat << endl;

  // use resize since out of place operation
  sums->resize (max_npart);
  zapmask->resize (max_npart);

  // reserve space to hold one more than current
  output->resize (min_output_ndat);
}

/* call set of transformations */
void dsp::SpectralKurtosis::transformation () try
{
  if (zero_DM && has_zero_DM_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation"
              " zero_DM_buffering_policy pre_transformation" << endl;

    get_zero_DM_buffering_policy()->pre_transformation();
  }

  if (!prepared)
    prepare();

  const uint64_t ndat  = input->get_ndat();
  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input ndat=" << ndat << endl;

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    resolution[ires].prepare (ndat);
    if (ires > 0)
      resolution[ires].compatible( resolution[ires-1] );
  }

  unsigned max_npart = resolution.front().get_npart();
  unsigned min_output_ndat = resolution.back().get_output_ndat();

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input max"
         << " npart=" << max_npart
         << " output_ndat=" << min_output_ndat << endl;

  if (has_buffering_policy())
  {
    if (verbose || debugd < 1)
      cerr << "dsp::SpectralKurtosis::transformation setting next_start_sample="
           << min_output_ndat << endl;

    get_buffering_policy()->set_next_start (min_output_ndat);

    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation set_next_start done" << endl;
  }

  if (zero_DM && has_zero_DM_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation zero_DM_buffering_policy set_next_start " << endl;

    get_zero_DM_buffering_policy()->set_next_start(min_output_ndat);
  }

  prepare_output ();

  // ensure output containers are sized correctly
  reserve();

  if ((ndat == 0) || (max_npart == 0))
    return;

  // perform SK functions
  if (verbose)
  {
    cerr << "dsp::SpectralKurtosis::transformation: calling compute" << endl;
    cerr << "dsp::SpectralKurtosis::transformation:: detect_time_freq=" << detect_time_freq
      << " detect_time=" << detect_time << " detect_freq=" << detect_freq << endl;
  }

  compute ();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::transformation: calling detect" << endl;

  if (report)
  {
    float_reporter.emit(
      "input",
      (float*) input->get_datptr(),
      input->get_nchan(),
      input->get_npol(),
      input->get_ndat(),
      input->get_ndim());

    float_reporter.emit(
      "sums",
      sums->get_dattfp(),
      sums->get_nchan(),
      sums->get_npol(),
      sums->get_ndat(),
      sums->get_ndim());
  }

  detect ();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::transformation: calling mask" << endl;

  if (report)
  {
    char_reporter.emit(
      "zapmask",
      zapmask->get_datptr(),
      zapmask->get_nchan(),
      zapmask->get_npol(),
      zapmask->get_ndat(),
      zapmask->get_ndim());
  }

  mask ();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::transformation: done" << endl;

  if (report)
  {
    float_reporter.emit(
      "output",
      output->get_datptr(),
      output->get_nchan(),
      output->get_npol(),
      output->get_ndat(),
      output->get_ndim());
  }

  //insertsk();
}
catch (Error& error)
{
  throw error += "dsp::SpectralKurtosis::transformation";
}

void dsp::SpectralKurtosis::compute () try
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::compute" << endl;

  const dsp::TimeSeries* compute_input = get_input();

  if (zero_DM)
  {
    compute_input = get_zero_DM_input();
    if (verbose)
    {
      cerr << "dsp::SpectralKurtosis::compute: using zero DM input" << endl;
      cerr << "dsp::SpectralKurtosis::compute: sample input=" << input->get_input_sample()
           << " zero_DM_input=" << get_zero_DM_input()->get_input_sample() << endl;
      cerr << "dsp::SpectralKurtosis::compute: ndate input=" << input->get_ndat()
           <<  " zero_DM_input=" << get_zero_DM_input()->get_ndat() << endl;
    }
    // compute_input->set_input_sample(input->get_input_sample());
  }

  const unsigned M = resolution.front().get_M();
  const unsigned npart = resolution.front().get_npart();
  const unsigned overlap_offset = resolution.front().get_overlap_offset();

  if (engine)
  {
    engine->compute (compute_input, sums, nullptr, M);
    return;
  }

  float* sum_dat = sums->get_dattfp();
  auto sum_wt = sums->get_weights();
  const unsigned sum_ndim = sums->get_ndim();

  assert (sum_ndim == 2);

  const bool input_detected = input->get_detected();
  const bool zero_is_valid = ! input->get_zeroed_data ();

  const float* indat = nullptr;

  auto order = compute_input->get_order();

  const unsigned int stride = (order == dsp::TimeSeries::OrderTFP) ? (nchan * npol * ndim) : (ndim);
  const unsigned int offset_nfloat = overlap_offset * ndim;

  for (unsigned ipart=0; ipart < npart; ipart++)
  {
    if (order == dsp::TimeSeries::OrderTFP)
      indat = compute_input->get_dattfp() + (overlap_offset * ipart * stride);

    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < sums_npol; ipol++)
      {
        if (order == dsp::TimeSeries::OrderFPT)
          indat = compute_input->get_datptr (ichan, ipol) + ipart * offset_nfloat;

        double S1_sum = 0;
        double S2_sum = 0;
        unsigned count = 0;

        if (input_detected)
        {
          for (unsigned i=0; i<M; i++)
          {
            float val = indat[stride*i];

            if (zero_is_valid || val != 0)
            {
              S1_sum += val;
              S2_sum += val * val;
              count ++;
            }
          }
        }
        else
        {
          // Square Law Detect for S1 + S2
          for (unsigned i=0; i<M; i++)
          {
            float re = indat[stride*i];
            float im = indat[stride*i+1];
            float sqld = (re * re) + (im * im);

            if (zero_is_valid || sqld != 0)
            {
              S1_sum += sqld;
              S2_sum += sqld * sqld;
              count ++;
            }
          }
        }

        unsigned out_index = ichan*sums_npol + ipol;

        // store the S1 and S2 sums for later SK calculation
        sum_dat[out_index*2] = S1_sum;
        sum_dat[out_index*2+1] = S2_sum;

        if (ipol == 0)
        {
          if (verbose && count != M)
            cerr << "ipart=" << ipart << "/" << npart << " chan=" << ichan << " ipol=" << ipol
                 << " count=" << count << " != " << M << endl;
          sum_wt[ichan] = count;
        }
        indat += ndim;
      }
    }
    sum_dat += nchan * sums_npol * sum_ndim;
    sum_wt += nchan;
  }

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::compute done" << endl;

  if (debugd < 1)
    debugd++;
}
catch (Error& error)
{
  throw error += "dsp::SpectralKurtosis::compute";
}

void dsp::SpectralKurtosis::set_thresholds (float _std_devs)
{
  for (unsigned ires=0; ires < resolution.size(); ires++)
    resolution[ires].set_std_devs (_std_devs);
}

void dsp::SpectralKurtosis::Resolution::set_thresholds (bool verbose) const
{
  if (verbose)
    std::cerr << "dsp::SpectralKurtosis::Resolution::set_thresholds SKlimits M=" << M << " Nd=" << Nd << " sigma=" << std_devs << endl;

  dsp::SKLimits limits(M, std_devs);
  limits.set_Nd(Nd);
  limits.calc_limits();

  thresholds.resize(2);
  thresholds[0] = (float) limits.get_lower_threshold();
  thresholds[1] = (float) limits.get_upper_threshold();

  if (verbose)
    std::cerr << "dsp::SpectralKurtosis::Resolution::set_thresholds "
         "M=" << M << " std_devs=" << std_devs  <<
         " [" << thresholds[0] << " - " << thresholds[1] << "]" << endl;
}

void dsp::SpectralKurtosis::set_channel_range (unsigned start, unsigned end)
{
  resolution[0].add_include( pair<unsigned,unsigned> (start, end-1) );
}

void dsp::SpectralKurtosis::detect () try
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect" << endl;

  if (verbose || debugd < 1)
  {
    cerr << "dsp::SpectralKurtosis::detect INPUT"
         << " nchan=" << nchan << " nbit=" << input->get_nbit()
         << " npol=" << npol << " ndim=" << ndim << endl;

    cerr << "dsp::SpectralKurtosis::detect OUTPUT"
         << " ndat=" << zapmask->get_ndat() << " nchan=" << zapmask->get_nchan()
         << " nbit=" << zapmask->get_nbit() << " npol=" << zapmask->get_npol()
         << " ndim=" << zapmask->get_ndim() << endl;
  }

  // reset the mask to all 0 (no zapping)
  reset_mask();

  if (report)
  {
    char_reporter.emit(
      "zapmask_tscr",
      zapmask->get_datptr(),
      zapmask->get_nchan(),
      zapmask->get_npol(),
      zapmask->get_ndat(),
      zapmask->get_ndim());
  }

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    if (ires > 0)
      tscrunch_sums (resolution[ires-1], resolution[ires]);

    // apply the SKFB estimates to the mask
    if (detect_time_freq)
      detect_skfb (ires);

    if (report)
    {
      char_reporter.emit(
      "zapmask_skfb",
      zapmask->get_datptr(),
      zapmask->get_nchan(),
      zapmask->get_npol(),
      zapmask->get_ndat(),
      zapmask->get_ndim());
    }

    if (detect_time)
      detect_fscr (ires);

    if (report)
    {
      char_reporter.emit(
      "zapmask_fscr",
      zapmask->get_datptr(),
      zapmask->get_nchan(),
      zapmask->get_npol(),
      zapmask->get_ndat(),
      zapmask->get_ndim());
    }

    if (ires == 0)
      count_zapped ();
  }

  // tscrunch the SKFB estimates and mask
  if (detect_freq)
    detect_tscr ();

  if (debugd < 1)
    debugd++;
}
catch (Error& error)
{
  throw error += "dsp::SpectralKurtosis::detect";
}

void dsp::SpectralKurtosis::tscrunch_sums (Resolution& from, Resolution& to) try
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::tscrunch_sums from "
      "M=" << from.get_M() << " noverlap=" << from.get_noverlap() << " to "
      "M=" << to.get_M() << " noverlap=" << to.get_noverlap() << endl;

  // double check that the tscrunch can be done
  to.compatible(from);

  const unsigned nsum = to.get_M() / from.get_M();
  const unsigned offset = to.get_overlap_offset() / from.get_overlap_offset();
  const uint64_t npart = to.get_npart();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::tscrunch_sums convert"
      " npart=" << npart << " nsum=" << nsum << " offset=" << offset << endl;

  unsigned sums_ndim = sums->get_ndim();
  assert (sums_ndim == 2);
  assert (sums->get_nchan() == nchan);
  assert (sums->get_npol() == sums_npol);

  float* outdat = sums->get_dattfp();
  auto outwt = sums->get_weights();

  // in place tscrunch
  const float* indat = outdat;
  auto inwt = outwt;

  const uint64_t fpd_blocksize = nchan * sums_npol * sums_ndim;
  const uint64_t input_offset = fpd_blocksize * offset;
  const uint64_t wt_offset = nchan * offset;

  const uint64_t last_idat = (npart-1)*offset+(nsum-1)*offset;

  if (verbose)
    cerr << "dsp::SpectralKurtosis::tscrunch_sums last_idat=" << last_idat << " input.ndat=" << sums->get_ndat() << endl;

  assert (last_idat < sums->get_ndat());
  assert (last_idat < sums->get_nweights());

  // compare SK estimator for each pol to expected values
  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    // for each channel and pol in the SKFB
    for (unsigned idat=0; idat < fpd_blocksize; idat++)
      outdat[idat] = indat[idat];

    for (unsigned ichan=0; ichan < nchan; ichan++)
      outwt[ichan] = inwt[ichan];

    for (unsigned isum=1; isum < nsum; isum++)
    {
      for (unsigned idat=0; idat < fpd_blocksize; idat++)
        outdat[idat] += indat[isum*input_offset+idat];

      for (unsigned ichan=0; ichan < nchan; ichan++)
        outwt[ichan] += inwt[isum*wt_offset+ichan];
    }

    indat += input_offset;
    inwt += wt_offset;

    outdat += fpd_blocksize;
    outwt += nchan;
  }

  sums->set_ndat(npart);

  if (verbose)
    cerr << "dsp::SpectralKurtosis::tscrunch_sums done" << endl;
}
catch (Error& error)
{
  throw error += "dsp::SpectralKurtosis::tscrunch_sums";
}

const dsp::SKLimits* dsp::SpectralKurtosis::get_limits (unsigned count, float std_devs)
{
  if (count >= dynamic_limits.size())
    dynamic_limits.resize(count+1, nullptr);

  if (!dynamic_limits[count])
  {
    auto limits = new dsp::SKLimits(count, std_devs);
    limits->set_Nd(integrated_Nd);
    limits->set_ntest(sums_npol);
    limits->calc_limits();
    dynamic_limits[count] = limits;

    if (verbose)
      cerr << "dsp::SpectralKurtosis::get_limits new limits for count=" << count
          << " " << limits->get_lower_threshold()
          << " " << limits->get_upper_threshold() << endl;
  }

  return dynamic_limits[count];
}

/*
 * Use the tscrunched SK statistic from the SKFB to detect RFI on each channel
 */
void dsp::SpectralKurtosis::detect_tscr () try
{
  unsigned M = resolution.back().get_M();
  unsigned npart = resolution.back().get_npart();
  unsigned noverlap = resolution.back().get_noverlap();
  float std_devs = resolution.back().get_std_devs();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_tscr M=" << M << " npart=" << npart << " noverlap=" << noverlap << endl;

  float* sumdat = sums->get_dattfp();
  auto sumwt = sums->get_weights();

  uint64_t stride = nchan * sums_npol;

  const vector<bool>& channels = resolution.front().get_channels(nchan);

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    if (!channels[ichan])
      continue;

    uint64_t count = 0;
    for (unsigned ipart=0; ipart < npart; ipart+=noverlap)
    {
      count += sumwt[ipart*nchan + ichan];
    }

    if (count == 0)
      continue;

    auto limits = get_limits (count, std_devs);
    float lower = limits->get_lower_threshold();
    float upper = limits->get_upper_threshold();

    float M_t = count;

    // Factor outside of brackets in Equation (8) of Nita & Gary (2010b)
    float M_fac = (M_t * integrated_Nd + 1) / (M_t - 1);

    if (verbose || debugd < 1)
      cerr << "dsp::SpectralKurtosis::compute tscr ichan=" << ichan << " M=" << M_t <<" M_fac=" << M_fac << endl;

    bool zap_chan = false;

    for (unsigned ipol=0; ipol < sums_npol; ipol++)
    {
      double S1 = 0.0;
      double S2 = 0.0;

      uint64_t offset = ichan*sums_npol + ipol;

      for (unsigned ipart=0; ipart < npart; ipart+=noverlap)
      {
        unsigned idx = ipart * stride + offset;
        S1 += sumdat[idx * 2];
        S2 += sumdat[idx * 2 + 1];
      }

      // Equation (8) of Nita & Gary (2010b)
      double SK = M_fac * (M_t * (S2 / (S1 * S1)) - 1);
      if (SK > upper || SK < lower)
        zap_chan = true;
    }

    SK_freq.tested ++;

    if (zap_chan)
    {
      SK_freq.zapped ++;

      // if (verbose)
      //   cerr << "dsp::SpectralKurtosis::detect_tscr zap V=" << V << ", "
      //        << "ichan=" << ichan << endl;
      unsigned char* outdat = zapmask->get_datptr();
      for (unsigned ipart=0; ipart < npart; ipart++)
      {
        outdat[ichan] = 1;
        outdat += nchan;
      }
    }
  }

  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_tscr done" << endl;
}
catch (Error& error)
{
  throw error += "dsp::SpectralKurtosis::detect_tscr";
}

void dsp::SpectralKurtosis::detect_skfb (unsigned ires) try
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_skfb(" << ires << ")" << endl;

  unsigned M = resolution[ires].get_M ();
  unsigned Mmin = resolution[0].get_M ();

  unsigned npart = resolution[ires].get_npart();
  float std_devs = resolution[ires].get_std_devs ();

  unsigned nflag = 1;
  unsigned flag_step = 1;
  unsigned flag_offset = 1;

  if (ires > 0)
  {
    nflag = M / Mmin;
    flag_step = resolution[0].get_noverlap();
    flag_offset = nflag * flag_step / resolution[ires].get_noverlap();
  }

  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_skfb nflag=" << nflag << " step=" << flag_step << endl;

  if (engine)
  {
    // TODO: Engine interface will need to be updated for adaptive SK
    engine->detect_ft (sums, zapmask, 0, 0);
    return;
  }

  float* sum_dat = sums->get_dattfp();
  auto sum_wt = sums->get_weights();

  unsigned char* outdat = zapmask->get_datptr();
  const unsigned sums_ndim = sums->get_ndim();
  assert (sums_ndim == 2);

  // count of tested and zapped
  Count count;

  // compare SK estimator for each pol to expected values
  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    // for each channel and pol in the SKFB
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      unsigned Mprime = sum_wt[ichan];

      if (Mprime == 0)
        continue;

      bool zap = true;

      if (Mprime > 2)
      {
        // Factor outside of brackets in Equation (8) of Nita & Gary (2010b)
        const float M_fac = (float)(Mprime*integrated_Nd+1) / (Mprime-1);
        auto limits = get_limits (Mprime, std_devs);
        float lower = limits->get_lower_threshold();
        float upper = limits->get_upper_threshold();

        zap = false;

        for (unsigned ipol=0; ipol < sums_npol; ipol++)
        {
          unsigned index = (sums_npol*ichan + ipol) * sums_ndim;
          float S1_sum = sum_dat[index];
          float S2_sum = sum_dat[index+1];

          float V = M_fac * (Mprime * (S2_sum / (S1_sum * S1_sum)) - 1);

          if (V > upper || V < lower)
            zap = true;
        }
      }

      count.tested ++;

      if (zap)
      {
        count.zapped ++;

        if (omit_outliers)
        {
          /* Set count, S1 and S2 for this outlier to zero so that it does not contribute to
            SK estimates for larger values M in tscrunch_sums */
          sum_wt[ichan] = 0;
          for (unsigned ipol=0; ipol < sums_npol; ipol++)
          {
            unsigned index = (sums_npol*ichan + ipol) * sums_ndim;
            sum_dat[index] = sum_dat[index+1] = 0.0;
          }
        }

        for (unsigned iflag=0; iflag < nflag; iflag++)
        {
          unsigned outdex = iflag*nchan*flag_step + ichan;
          outdat[outdex] = 1;
        }
      }
    }

    sum_dat += nchan * sums_npol * sums_ndim;
    sum_wt += nchan;
    outdat += nchan * flag_offset;
  }

  resolution[ires].increment_time_freq(count);

  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_skfb done" << endl;
}
catch (Error& error)
{
  throw error += "dsp::SpectralKurtosis::detect_skfb ires=" + tostring(ires) + " M=" + tostring(resolution[ires].get_M());
}

void dsp::SpectralKurtosis::reset_mask ()
{
  if (engine)
  {
    engine->reset_mask (zapmask);
    return;
  }

  zapmask->zero();
}

void dsp::SpectralKurtosis::count_zapped ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::count_zapped hits=" << unfiltered_hits << endl;

  const float * indat;
  unsigned char * outdat;

  if (engine)
  {
    indat = engine->get_estimates (sums);
    outdat = engine->get_zapmask(zapmask);
  }
  else
  {
    indat  = sums->get_dattfp();
    outdat = zapmask->get_datptr();
  }

  const unsigned sums_ndim = sums->get_ndim();
  assert (sums_ndim == 2);

  unsigned ires = 0;
  unsigned M = resolution[ires].get_M();
  unsigned npart = resolution[ires].get_npart();

  // Factor outside of brackets in Equation (8) of Nita & Gary (2010b)
  const float M_fac = (float)(M*integrated_Nd+1) / (M-1);

  assert (npart == sums->get_ndat());
  if (unfiltered_hits == 0)
  {
    filtered_sum.resize (sums_npol * nchan);
    std::fill (filtered_sum.begin(), filtered_sum.end(), 0);

    filtered_hits.resize (nchan);
    std::fill (filtered_hits.begin(), filtered_hits.end(), 0);

    unfiltered_sum.resize (sums_npol * nchan);
    std::fill (unfiltered_sum.begin(), unfiltered_sum.end(), 0);
  }

  const vector<bool>& channels = resolution[ires].get_channels(nchan);

  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    unfiltered_hits ++;

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      if (!channels[ichan])
        continue;

      for (unsigned ipol=0; ipol < sums_npol; ipol++)
      {
        uint64_t index = ((ipart*nchan + ichan) * sums_npol + ipol) * sums_ndim;
        unsigned outdex = ichan * sums_npol + ipol;

        float S1_sum = indat[index];
        float S2_sum = indat[index+1];

        float V = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);

        unfiltered_sum[outdex] += V;

        if (outdat[(ipart*nchan) + ichan] != 1)
          filtered_sum[outdex] += V;
      }

      filtered_hits[ichan] ++;
    }
  }

  if (verbose)
    cerr << "dsp::SpectralKurtosis::count_zapped done" << endl;
}

void dsp::SpectralKurtosis::detect_fscr (unsigned ires) try
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_fscr(" << ires << ")" << endl;

  unsigned M = resolution[ires].get_M();
  unsigned npart = resolution[ires].get_npart();
  float std_devs = resolution[ires].get_std_devs();

  if (engine)
  {
    // float one_sigma_idat   = sqrt(mu2 / (float) nchan);
    // const float upper = 1 + ((1+std_devs) * one_sigma_idat);
    // const float lower = 1 - ((1+std_devs) * one_sigma_idat);
    // cerr << "dsp::SpectralKurtosis::detect_fscr:" <<
    //   " upper=" << upper <<
    //   " lower=" << lower << endl;
    // engine->detect_fscr (sums, zapmask, lower, upper, channels[0], channels[1]);

    /*
      NZAPP-207: WvS 2020-05-06 GPU interface needs to be updated to receive
      a mask of bool for each channel, instead of a single start/end range
    */
    unsigned ichan_start = 0;
    unsigned ichan_end = nchan;
    double mu2 = 0; // TODO: redefine for adaptive
    engine->detect_fscr (sums, zapmask, mu2, std_devs, ichan_start, ichan_end);

    return;
  }

  const float* indat = sums->get_dattfp();
  auto inwt = sums->get_weights();
  unsigned char* outdat = zapmask->get_datptr();

  const unsigned sums_ndim = sums->get_ndim();
  assert (sums_ndim == 2);

  unsigned zap_ipart;
  uint64_t nzap = 0;

  unsigned nflag = 1;
  unsigned flag_step = 1;
  unsigned flag_offset = 1;

  if (ires > 0)
  {
    nflag = M / resolution[0].get_M();
    flag_step = resolution[0].get_noverlap();
    flag_offset = nflag * flag_step / resolution[ires].get_noverlap();
  }

  const vector<bool>& channels = resolution[ires].get_channels(nchan);

  // count of tested and zapped
  Count count;

  // foreach SK integration
  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    zap_ipart = 0;
    for (unsigned ipol=0; ipol < sums_npol; ipol++)
    {
      double sk_avg = 0;
      double M_tot = 0;
      double Msq_tot = 0;
      double chan_count = 0;

      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        if (channels[ichan] && (!omit_outliers || outdat[ichan] == 0))
        {
          unsigned index = (sums_npol*ichan + ipol) * sums_ndim;
          float S1_sum = indat[index];
          float S2_sum = indat[index+1];

          unsigned Mprime = inwt[ichan];
          // Factor outside of brackets in Equation (8) of Nita & Gary (2010b)
          const float M_fac = (float)(Mprime*integrated_Nd+1) / (Mprime-1);
          float V = M_fac * (Mprime * (S2_sum / (S1_sum * S1_sum)) - 1);

          // compute a weighted average SK
          sk_avg += V * Mprime;
          M_tot += Mprime;
          Msq_tot += Mprime * Mprime;
          chan_count ++;
        }
      }

      if (chan_count > 0)
      {
        sk_avg /= M_tot;

        double M_mean = M_tot / chan_count;
        double Msq_mean = Msq_tot / chan_count;

        // Kish's design effect
        double Deff = Msq_mean / (M_mean * M_mean);
        double effective_count = chan_count / Deff;

        // add 0.5 to round to nearest integer
        auto limits = get_limits (M_mean + 0.5, std_devs);

        // reduce threshold to account for averaging over an effective number of channel
        float threshold = limits->get_symmetric_threshold() / sqrt(effective_count);

        float avg_upper_thresh = 1.0 + threshold;
        float avg_lower_thresh = 1.0 - threshold;

        if ((sk_avg > avg_upper_thresh) || (sk_avg < avg_lower_thresh))
        {
          if (verbose)
            cerr << "Zapping ipart=" << ipart << " ipol=" << ipol << " sk_avg=" << sk_avg
                 << " [" << avg_lower_thresh << " - " << avg_upper_thresh
                 << "] M=" << M << " mean_M=" << M_mean << " chan=" << chan_count << " effective_chan=" << effective_count << endl;

          zap_ipart = 1;
        }
      }
    }

    count.tested ++;

    if (zap_ipart)
    {
      count.zapped ++;

      for (unsigned iflag=0; iflag < nflag; iflag++)
      {
        unsigned outdex = iflag*nchan*flag_step;
        for (unsigned ichan=0; ichan<nchan; ichan++)
          outdat[outdex + ichan] = 1;
      }

      nzap += nchan;
    }

    indat += nchan * sums_npol * sums_ndim;
    inwt += nchan;
    outdat += nchan * flag_offset;
  }

  resolution[ires].increment_time(count);

  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_fscr times zapped=" << count.zapped << endl;
}
catch (Error& error)
{
  throw error += "dsp::SpectralKurtosis::detect_fscr";
}

//! Perform the transformation on the input time series
void dsp::SpectralKurtosis::mask () try
{
  // indicate the output timeseries contains zeroed data
  output->set_zeroed_data (true);

  // resize the output to ensure the hits array is reallocated
  if (engine)
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::mask output->resize(" << output->get_ndat() << ")" << endl;

    output->resize (output->get_ndat());
  }

  // get base pointer to mask bitseries
  unsigned char * mask = zapmask->get_datptr ();

  unsigned M = resolution.front().get_M();
  // unsigned overlap = resolution.front().get_noverlap();
  unsigned overlap_offset = resolution.front().get_overlap_offset();

  // NZAPP-208 WvS: the most samples are lost at the most coarse resolution
  unsigned max_M = resolution.back().get_M();
  unsigned max_overlap_offset = resolution.back().get_overlap_offset();
  unsigned idat_start = max_M - max_overlap_offset;

#if _DEBUG
  cerr << "front.ndat=" << resolution.front().get_output_ndat() << endl;
  cerr << "back.ndat=" << resolution.back().get_output_ndat() << endl;
#endif

  // the number of fine blocks to skip
  unsigned nskip = idat_start / overlap_offset;
  // the number of fine blocks to process
  unsigned npart = (resolution.back().get_output_ndat() - M) / overlap_offset + 1;

#if _DEBUG
  cerr << "front.get_npart()=" << resolution.front().get_npart() << endl;
  cerr << "back.get_npart()=" << resolution.back().get_npart() << endl;
  cerr << "npart=" << npart << " nskip=" << nskip << endl;
#endif

  assert ( idat_start % overlap_offset == 0 );
  assert ( (resolution.back().get_output_ndat() - M) % overlap_offset == 0 );
  assert ( npart+nskip <= resolution.front().get_npart() );

  if ( idat_start + resolution.back().get_output_ndat() > input->get_ndat() )
    throw Error (InvalidState, "dsp::SpectralKurtosis::mask",
                 "idat_start=%u + output.ndat=%u > input.ndat=%u",
                 idat_start, resolution.back().get_output_ndat(), input->get_ndat());

  assert ( resolution.back().get_output_ndat() == output->get_ndat() );

  if (engine)
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::mask engine->setup(" << nchan << ")" << endl;

    engine->mask (zapmask, input, output, M);
  }
  else
  {
    // mask is a TFP ordered bit series, output is FPT order TimeSeries
    const unsigned nfloat = M * ndim;
    const unsigned int offset_nfloat = overlap_offset * ndim;

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        const float * indat  = input->get_datptr(ichan, ipol) + idat_start;
        float * outdat = output->get_datptr(ichan, ipol);
        for (uint64_t ipart=0; ipart < npart; ipart++)
        {
          if (mask[(ipart+nskip)*nchan+ichan])
          {
            for (unsigned j=0; j<nfloat; j++)
              outdat[j] = 0;
          }
          else
          {
            for (unsigned j=0; j<nfloat; j++)
              outdat[j] = indat[j];
          }

          indat += offset_nfloat;
          outdat += offset_nfloat;
        }
      }
    }
  }

  if (debugd < 1)
    debugd++;

  if (verbose)
    cerr << "dsp::SpectralKurtosis::mask done" << endl;
}
catch (Error& error)
{
  throw error += "dsp::SpectralKurtosis::mask";
}

//!
void dsp::SpectralKurtosis::insertsk ()
{
  if (engine)
    engine->insertsk (sums, output, resolution.front().get_M());
}

