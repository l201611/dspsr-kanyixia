/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Staten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DynamicFilter.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Input.h"

#include "FTransform.h"
#include "malloc16.h"

#include <fstream>
#include <cstring>
#include <cassert>

using namespace std;

dsp::DynamicFilter::DynamicFilter (Pulsar::DynamicResponse* _response)
{
  dynamic_response = _response;
}

//! Use zero-padded inverse Fourier transform to interpolate between samples
void zero_pad_interpolate (complex<float>* dest, unsigned n_dest, const complex<float>* src, unsigned n_src, unsigned n_negative = 0, const char* filename = 0)
{
  if (n_src >= n_dest)
    throw Error (InvalidParam, "zero_pad_interpolate",
                "n_src=%u >= n_dest=%u", n_src, n_dest);

  assert (dest != nullptr);
  assert (src != nullptr);
  assert (n_dest > 0);
  assert (n_src > 0);

  Array16<float> dom2 (n_dest * 2);
  auto c_dom2 = reinterpret_cast<complex<float>*>(dom2.get());

  FTransform::bcc1d (n_src, dom2, reinterpret_cast<const float*>(src));

  unsigned zero_start = n_src - n_negative;
  unsigned zero_end = n_dest - n_negative;

  if (n_negative)
  {
    // shift the negative lags to the end of the array
    auto dest = c_dom2 + zero_end;
    auto src = c_dom2 + zero_start;
    auto n_bytes = n_negative * sizeof(complex<float>);
    memmove(dest, src, n_bytes);
  }

  // zero pad the rest
  for (unsigned ipt=zero_start; ipt<zero_end; ipt++)
    c_dom2[ipt] = 0;

  if (filename)
  {
    ofstream os (filename);
    for (unsigned i=0; i<n_src + 10; i++)
        os << c_dom2[i].real() << " " << c_dom2[i].imag() << " " << std::abs(c_dom2[i]) << endl;
  }

  FTransform::fcc1d (n_dest, reinterpret_cast<float*>(dest), dom2);

  float scalefac = 1.0;

  if (FTransform::get_norm() == FTransform::unnormalized)
    scalefac = 1.0 / float(n_src);
  else
    scalefac = float(n_dest) / float(n_src);

  for (unsigned ipt=0; ipt < n_dest; ipt++)
    dest[ipt] *= scalefac;
}

void dsp::DynamicFilter::build (const Observation* input)
{
  unsigned ntime = dynamic_response->get_ntime();
  unsigned native_ndat = dynamic_response->get_nchan();
  unsigned required_ndat = get_nchan() * get_ndat();

  // calculate the complex response of the scalar
  auto phasors = reinterpret_cast< complex<float>* > ( buffer );

  auto data = dynamic_response->get_data();
  unsigned offset = current_itime * native_ndat;

  if (verbose)
    cerr << "dsp::DynamicFilter::build current_itime=" << current_itime << " ntime=" << ntime << endl;

  if (native_ndat < required_ndat)
  {
    if (verbose)
      cerr << "dsp::DynamicFilter::build interpolate required_ndat=" << required_ndat << " > native_ndat=" << native_ndat << endl;

    tmp_in.resize(native_ndat);

    for (unsigned ipt=0; ipt<native_ndat; ipt++)
      tmp_in[ipt] = data[ipt + offset];

    const char* fname = nullptr;

    if (verbose)
    {
      string filename = "zero_padded_" + tostring(current_itime) + ".txt";
      fname = filename.c_str();
    }

    zero_pad_interpolate(phasors, required_ndat, tmp_in.data(), native_ndat, native_ndat-1, fname);
  }
  else
  {
    for (unsigned ipt=0; ipt<required_ndat; ipt++)
      phasors[ipt] = data[ipt + offset];
  }

  whole_swapped = true;
}

class overlap
{
  double between = 0;  // end time of A precedes start time of B

  public:

  //! Return the number of seconds by which A overlaps B
  /*! If the overlap is zero, and the end of A precedes the start of B, 
      then between set to the negative number of seconds in the gap between the intervals.
      If the overlap is zero, and the start of A follows the end of B, 
      then between set to the positive number of seconds in the gap between the intervals.
  */
  double overlap_seconds (const MJD& A_start, const MJD& A_end, const MJD& B_start, const MJD& B_end)
  {
    if (A_end < B_start)
    {
      between = -(B_start - A_end).in_seconds();
      return 0;
    }
    if (B_end < A_start)
    {
      between = (A_start - B_end).in_seconds();
      return 0;
    }

    // minimum overlap
    double overlap_1 = (B_end - A_start).in_seconds();
    double overlap_2 = (A_end - B_start).in_seconds();
    double overlap = std::min(overlap_1, overlap_2);

    // minimum duration
    double duration_A = (A_end - A_start).in_seconds();
    double duration_B = (B_end - B_start).in_seconds();
    double duration = std::min(duration_A, duration_B);

    // minimum of duration and overlap
    return std::min(overlap, duration);
  }

  //! Return the number of seconds in the gap between the intervals, as last computed by overlap_seconds
  double get_between() const { return between; }
};

void dsp::DynamicFilter::configure (const Observation* input, unsigned nchan)
{
  if (verbose)
    cerr << "dsp::DynamicFilter::configure nchan=" << nchan << " ndat=" << get_ndat() << endl;

  if (input != current_input)
  {
    double input_bw = input->get_bandwidth();
    double input_cfreq = input->get_centre_frequency();

    double ext_bw = dynamic_response->get_bandwidth();
    double ext_cfreq = dynamic_response->get_centre_frequency();

    double one_Hz = 1e-6; // in MHz
    if (fabs(input_bw - ext_bw) > one_Hz)
    {
        throw Error (InvalidState, "dsp::DynamicFilter::configure", 
            "filter bandwidth= %lf != input bandwidth=%lf", ext_bw, input_bw);
    }
    if (fabs(input_cfreq - ext_cfreq) > one_Hz)
    {
        throw Error (InvalidState, "dsp::DynamicFilter::configure", 
            "filter centre frequency= %lf != input centre frequency=%lf", ext_cfreq, input_cfreq);
    }
  }

  current_input = input;

  MJD input_start_time = input->get_start_time();
  MJD input_end_time = input->get_end_time();

  impulse_pos = dynamic_response->get_impulse_pos();
  impulse_neg = dynamic_response->get_impulse_neg();

  if (input_start_time > current_start_time && input_end_time < current_end_time)
  {
    // the observation is spanned by the current time slice, no change
    return;
  }

  unsigned ntime = dynamic_response->get_ntime();
  MJD min_epoch = dynamic_response->get_minimum_epoch ();
  MJD max_epoch = dynamic_response->get_maximum_epoch ();

  double interval = (max_epoch - min_epoch).in_seconds() / ntime;

  if (verbose)
    cerr << "dsp::DynamicFilter::configure ntime=" << ntime << " interval=" << interval << " seconds" << endl;

  overlap over;

  double max_overlap = 0.0;
  double min_between = 0.0;
  unsigned min_between_itime = 0;
  unsigned best_itime = current_itime;

  for (unsigned itime = best_itime; itime < ntime; itime++)
  {
    MJD interval_start_time = min_epoch + interval * itime;
    MJD interval_end_time = interval_start_time + interval;

    double overlap = over.overlap_seconds (input_start_time, input_end_time, interval_start_time, interval_end_time);

    if (overlap > 0 && verbose)
    {
      cerr << "dsp::DynamicFilter::configure itime=" << itime << " overlap=" << overlap << endl;
    }

    if (overlap > max_overlap)
    {
      if (verbose)
        cerr << "dsp::DynamicFilter::configure max overlap old=" << max_overlap << " new=" << overlap << endl;

      max_overlap = overlap;
      best_itime = itime;
    }
    else if (overlap == 0)
    {
      double between = over.get_between();

      if (verbose)
        cerr << "dsp::DynamicFilter::configure itime=" << itime << " between=" << between << endl;

      if (between < min_between)
      {
        if (verbose)
          cerr << "dsp::DynamicFilter::configure min between old=" << min_between << " new=" << between << endl;

        min_between = between;
        min_between_itime = itime;
      }

      // it will only get more negative
      if (between < 0)
        break;
    }
  }

  if (max_overlap == 0.0)
  {
    if (verbose)
      cerr << "dsp::DynamicFilter::configure no overlapping interval"
              " - min between=" << min_between << " at " << min_between_itime << endl;
    best_itime = min_between_itime;
  }
  else if (verbose)
  {
    cerr << "dsp::DynamicFilter::configure best overlapping interval"
            " - max overlap=" << max_overlap << " at " << best_itime << endl;
  }

  MJD zero;
  if (current_start_time == zero || current_itime != best_itime)
  {
    cerr << "dsp::DynamicFilter::configure update current_itime old=" << current_itime << " new=" << best_itime << endl;
    current_start_time = min_epoch + interval * best_itime;
    current_end_time = current_start_time + interval;
    current_itime = best_itime;
    changed();
  }
}
