/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/RescaleMeanStdCalculator.h"

#include <assert.h>
#include <algorithm>

using namespace std;

template<typename T>
void zero (vector<T>& data)
{
  std::fill(data.begin(), data.end(), 0);
}

void dsp::RescaleMeanStdCalculator::init(const dsp::TimeSeries* input, uint64_t nsample, bool output_time_total)
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleMeanStdCalculator::init nsample=" << nsample << endl;  npol  = input->get_npol();

  nchan = input->get_nchan();
  ndim  = input->get_ndim();

  if (output_time_total)
    time_total.resize (npol);

  freq_total.resize (npol);
  freq_totalsq.resize (npol);

  scale.resize (npol);
  offset.resize (npol);

  nintegrated = 0;
  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    if (output_time_total)
    {
      time_total[ipol].resize (nsample);
      zero (time_total[ipol]);
    }

    freq_total[ipol].resize (nchan);
    zero (freq_total[ipol]);

    freq_totalsq[ipol].resize (nchan);
    zero (freq_total[ipol]);

    scale[ipol].resize (nchan);
    offset[ipol].resize (nchan);
  }
}

uint64_t dsp::RescaleMeanStdCalculator::sample_data(const dsp::TimeSeries* input, uint64_t start_dat, uint64_t end_dat, bool output_time_total)
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleMeanStdCalculator::sample_data"
      << " start_dat=" << start_dat
      << ", end_dat=" << end_dat
      << ", output_time_total=" << output_time_total
      << endl;

  const bool input_voltage = (input->get_state() == Signal::Nyquist || input->get_state() == Signal::Analytic);
  uint64_t samp_dat = nintegrated;

  switch (input->get_order())
  {
  case TimeSeries::OrderTFP:
  {
    const float* in_data = input->get_dattfp();
    in_data += start_dat * nchan * npol * ndim;
    for (uint64_t idat=start_dat; idat < end_dat; idat++, samp_dat++)
    {
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        for (unsigned ipol=0; ipol < npol; ipol++)
        {
          for (unsigned idim=0; idim < ndim; idim++)
          {
            const double val = static_cast<double>(*in_data);
            freq_total[ipol][ichan] += val;
            freq_totalsq[ipol][ichan] += val * val;
            if (output_time_total)
            {
              if (input_voltage)
                time_total[ipol][samp_dat] += static_cast<float>(val * val);
              else
                time_total[ipol][samp_dat] += static_cast<float>(val);
            }
            in_data++;
          }
        }
      }
    }
    break;
  }
  case TimeSeries::OrderFPT:
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        const float* in_data = input->get_datptr (ichan, ipol);

        samp_dat = nintegrated;

        double sum = 0.0;
        double sumsq = 0.0;

        for (uint64_t idat=start_dat; idat < end_dat; idat++, samp_dat++)
        {
          for (unsigned idim=0; idim<ndim; idim++)
          {
            const double val = double(*in_data);
            sum += val;
            sumsq += val * val;

            if (output_time_total)
            {
              if (input_voltage)
                time_total[ipol][samp_dat] += static_cast<float>(val * val);
              else
                time_total[ipol][samp_dat] += static_cast<float>(val);
            }
            in_data++;
          }
        }
        freq_total[ipol][ichan] += sum;
        freq_totalsq[ipol][ichan] += sumsq;
      }
    }
    break;
  }
  default:
    throw Error (InvalidState, "dsp::Rescale::operate",
        "Requires data in TFP or FPT order");
  }

  nintegrated += (end_dat - start_dat);
  return nintegrated;
}

void dsp::RescaleMeanStdCalculator::compute()
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleMeanStdCalculator::compute" << endl;

  const double nval = static_cast<double>(nintegrated * ndim);
  const double nval_recip = (nval > 0) ? 1.0 / nval : 0;

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      double mean = freq_total[ipol][ichan] * nval_recip;
      double meansq = freq_totalsq[ipol][ichan] * nval_recip;
      double variance = meansq - mean*mean;
      offset[ipol][ichan] = -mean;
      scale[ipol][ichan] = (variance > 0) ? 1.0 / sqrt(variance) : 1.0;
    }
  }
}

void dsp::RescaleMeanStdCalculator::reset_sample_data()
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleMeanStdCalculator::reset_sample_data" << endl;

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    zero (freq_total[ipol]);
    zero (freq_totalsq[ipol]);
    if (ipol < time_total.size())
      zero (time_total[ipol]);
  }
  nintegrated = 0;
}

const double* dsp::RescaleMeanStdCalculator::get_mean (unsigned ipol) const
{
  assert (ipol < freq_total.size());
  return &(freq_total[ipol][0]);
}

const double* dsp::RescaleMeanStdCalculator::get_variance (unsigned ipol) const
{
  assert (ipol < freq_totalsq.size());
  return &(freq_totalsq[ipol][0]);
}
