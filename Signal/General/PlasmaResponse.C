/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "dsp/PlasmaResponse.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "ThreadContext.h"
#include "Error.h"

#include <complex>

using namespace std;

float dsp::PlasmaResponse::smearing_buffer = 0.1;

//! Set the centre frequency of the band-limited signal in MHz
void dsp::PlasmaResponse::set_centre_frequency (double _centre_frequency)
{
  if (centre_frequency != _centre_frequency)
    changed();

  centre_frequency = _centre_frequency;
}

//! Returns the centre frequency of the specified channel in MHz
double dsp::PlasmaResponse::get_centre_frequency (int ichan) const
{
  throw Error (InvalidState,
	       "PlasmaResponse::get_centre_frequency (ichan)"
	       "not implemented");
}

//! Set the bandwidth of the signal in MHz
void dsp::PlasmaResponse::set_bandwidth (double _bandwidth)
{
  if (bandwidth != _bandwidth)
    changed();

  bandwidth = _bandwidth;
}

//! Set the Doppler shift due to the Earth's motion
void dsp::PlasmaResponse::set_Doppler_shift (double _Doppler_shift)
{
  if (Doppler_shift != _Doppler_shift)
    changed();

  Doppler_shift = _Doppler_shift;
}

void dsp::PlasmaResponse::configure (const Observation* input, unsigned channels)
{
  input_nchan = input->get_nchan();
  oversampling_factor = input->get_oversampling_factor();

  if (verbose)
    cerr << "dsp::PlasmaResponse::configure input.nchan=" << input->get_nchan()
	 << " channels=" << channels << "\n\t"
      " centre frequency=" << input->get_centre_frequency() <<
      " bandwidth=" << input->get_bandwidth () <<
      " dispersion measure=" << input->get_dispersion_measure() << endl;

  set_centre_frequency ( input->get_centre_frequency() );
  set_bandwidth ( input->get_bandwidth() );
  set_dc_centred ( input->get_dc_centred() );

  frequency_input.resize( input->get_nchan() );
  bandwidth_input.resize( input->get_nchan() );

  for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
  {
    frequency_input[ichan] = input->get_centre_frequency( ichan );
    bandwidth_input[ichan] = input->get_bandwidth() / input->get_nchan();
  }

  if (!channels)
    channels = input->get_nchan();

  set_nchan (channels);
  prepare();
}

unsigned smearing_samples_threshold = 16 * 1024 * 1024;

void dsp::PlasmaResponse::prepare ()
{
  unsigned threshold = smearing_samples_threshold / nchan;
  supported_channels = vector<bool> (nchan, true);
  unsigned ichan = 0;

  while( (impulse_neg = smearing_samples (-1)) > threshold )
  {
    supported_channels[ichan] = false;
    ichan ++;
    if (ichan == nchan)
      throw Error (InvalidState,
            "dsp::PlasmaResponse::prepare",
            "smearing samples=%u exceeds threshold=%u",
            impulse_neg, threshold);
  }

  if (verbose)
    cerr << "dsp::PlasmaResponse::prepare " << ichan << " unsupported channels" << endl;

  impulse_pos = smearing_samples (1);

  if (psrdisp_compatible)
  {
    cerr << "dsp::PlasmaResponse::prepare psrdisp compatibility\n"
      "   using symmetric impulse response function" << endl;
    impulse_pos = impulse_neg;
  }
}


/*!
  return x squared
  */
template <typename T> inline T sqr (T x) { return x*x; }

/*
  \param cfreq centre frequency, in MHz
  \param bw bandwidth, in MHz
  \retval dispersion smearing time across the specified band, in seconds
*/
double dsp::PlasmaResponse::smearing_time (double cfreq, double bw) const
{
  return delay_time (cfreq - fabs(0.5*bw), cfreq + fabs(0.5*bw));
}

double dsp::PlasmaResponse::delay_time (double freq1, double freq2) const
{
  return delay_time (freq1) - delay_time (freq2);
}

double dsp::PlasmaResponse::get_effective_smearing_time () const
{
  return smearing_time (0);
}

//! Return the effective number of smearing samples
unsigned dsp::PlasmaResponse::get_effective_smearing_samples () const
{
  if (verbose)
    cerr << "dsp::PlasmaResponse::get_effective_smearing_samples" << endl;

  return smearing_samples (0);
}

/*!
  Calculate the smearing time over the band (or the sub-band with
  the lowest centre frequency) in seconds.  This will determine the
  number of points "nsmear" that must be thrown away for each FFT.
*/
double dsp::PlasmaResponse::smearing_time (int half) const
{
  if ( ! (half==0 || half==-1 || half == 1) )
    throw Error (InvalidParam, "dsp::PlasmaResponse::smearing_time",
		 "invalid half=%d", half);

  double abs_bw = fabs (bandwidth);
  double ch_abs_bw = abs_bw / double(nchan);
  double lower_ch_cfreq = centre_frequency - (abs_bw - ch_abs_bw) / 2.0;

  unsigned ichan=0;
  while (ichan < supported_channels.size() && !supported_channels[ichan])
  {
    lower_ch_cfreq += ch_abs_bw;
    ichan++;
  }

  // calculate the smearing (in the specified half of the band)
  if (half)
  {
    ch_abs_bw /= 2.0;
    lower_ch_cfreq += double(half) * ch_abs_bw;
  }

  if (verbose)
    cerr << "dsp::PlasmaResponse::smearing_time freq=" << lower_ch_cfreq
         << " bw=" << ch_abs_bw << endl;

  double tsmear = smearing_time (lower_ch_cfreq, ch_abs_bw);

  if (verbose)
  {
    string band = "band";
    if (nchan>1)
      band = "worst channel";

    string side;
    if (half == 1)
      side = "upper half of the ";
    else if (half == -1)
      side = "lower half of the ";

    cerr << "dsp::PlasmaResponse::smearing_time in the " << side << band << ": "
         << float(tsmear*1e3) << " ms" << endl;
  }

  return tsmear;
}

unsigned dsp::PlasmaResponse::smearing_samples (int half) const
{
  double tsmear = smearing_time (half);
  double sampling_rate = get_sampling_rate();

  if (verbose)
    cerr << "dsp::PlasmaResponse::smearing_samples = "
         << int64_t(tsmear * sampling_rate) << endl;

  // add another ten percent, just to be sure that the pollution due
  // to the cyclical convolution effect is minimized
  if (psrdisp_compatible)
  {
     cerr << "dsp::PlasmaResponse::prepare psrdisp compatibility\n"
       "   increasing smearing time by 5 instead of "
          << smearing_buffer*100.0 << " percent" << endl;
    tsmear *= 1.05;
  }
  else
    tsmear *= (1.0 + smearing_buffer);

  // smear across one channel in number of time samples.
  unsigned nsmear = unsigned (ceil(tsmear * sampling_rate));

  if (psrdisp_compatible)
  {
    cerr << "dsp::PlasmaResponse::prepare psrdisp compatibility\n"
      "   rounding smear samples down instead of up" << endl;
     nsmear = unsigned (tsmear * sampling_rate);
  }

  if (verbose)
  {
    // recalculate the smearing time simply for display of new value
    tsmear = double (nsmear) / sampling_rate;
    cerr << "dsp::PlasmaResponse::smearing_samples effective smear time: "
         << tsmear*1e3 << " ms (" << nsmear << " pts)." << endl;
  }

  return nsmear;
}

//! Return the time spanned by samples dropped from the beginning of each cyclical convolution result
double dsp::PlasmaResponse::delay_time_pos () const
{
  return impulse_pos / get_sampling_rate(); 
}

//! Return the time spanned by samples dropped from the end of each cyclical convolution result
double dsp::PlasmaResponse::delay_time_neg () const
{
  return impulse_neg / get_sampling_rate();
}

//! Return the sampling rate in Hz of the resulting complex time samples
double dsp::PlasmaResponse::get_sampling_rate() const
{
  double ch_abs_bw = fabs (bandwidth) / double (nchan);
  return ch_abs_bw * 1e6;
}
