/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/RescaleMedianMadCalculator.h"

#include <assert.h>
#include <algorithm>

using namespace std;

template<typename T>
void zero (vector<T>& data)
{
  std::fill(data.begin(), data.end(), 0);;
}

void dsp::RescaleMedianMadCalculator::init(const dsp::TimeSeries* input, uint64_t nsample, bool output_time_total)
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleMedianMadCalculator::init nsample=" << nsample << endl;

  npol  = input->get_npol();
  nchan = input->get_nchan();
  ndim  = input->get_ndim();

  if (output_time_total)
    time_total.resize (npol);

  _sample_data.resize (npol);
  absolute_deviation.resize (npol);

  scale.resize (npol);
  offset.resize (npol);
  mean.resize (npol);
  variance.resize (npol);

  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    if (output_time_total)
    {
      time_total[ipol].resize (nsample);
      zero (time_total[ipol]);
    }

    scale[ipol].resize (nchan);
    zero (scale[ipol]);
    offset[ipol].resize (nchan);
    zero (offset[ipol]);

    mean[ipol].resize (nchan);
    zero (mean[ipol]);
    variance[ipol].resize (nchan);
    zero (variance[ipol]);

    _sample_data[ipol].resize (nchan);
    absolute_deviation[ipol].resize (nchan);
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      _sample_data[ipol][ichan].resize(nsample * ndim);
      zero (_sample_data[ipol][ichan]);
      absolute_deviation[ipol][ichan].resize(nsample * ndim);
      zero (absolute_deviation[ipol][ichan]);
    }
  }
}

uint64_t dsp::RescaleMedianMadCalculator::sample_data(const dsp::TimeSeries* input, uint64_t start_dat, uint64_t end_dat, bool output_time_total)
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleMedianMadCalculator::sample_data"
      << " start_dat=" << start_dat
      << ", end_dat=" << end_dat
      << ", output_time_total=" << output_time_total
      << endl;

  const bool input_voltage = (input->get_state() == Signal::Nyquist || input->get_state() == Signal::Analytic);

  switch (input->get_order())
  {
  case TimeSeries::OrderTFP:
  {
    const float* in_data = input->get_dattfp();
    in_data += start_dat * nchan * npol * ndim;
    uint64_t samp_dat = 0;
    for (uint64_t idat=start_dat; idat < end_dat; idat++, samp_dat++)
    {
      uint64_t odat = nintegrated + samp_dat;
      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        for (unsigned ipol=0; ipol < npol; ipol++)
        {
          for (unsigned idim=0; idim < ndim; idim++)
          {
            auto val = _sample_data[ipol][ichan][odat * ndim + idim] = *in_data;

            if (output_time_total)
            {
              if (input_voltage)
                time_total[ipol][odat] += val * val;
              else
                time_total[ipol][odat] += val;
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
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        const float* in_data = input->get_datptr (ichan, ipol);

        // ensure we offset by the starting time sample
        in_data += (start_dat * ndim);

        uint64_t samp_dat = 0;
        for (uint64_t idat=start_dat; idat < end_dat; idat++, samp_dat++)
        {
          uint64_t odat = nintegrated + samp_dat;
          for (unsigned idim=0; idim<ndim; idim++)
          {
            auto val = _sample_data[ipol][ichan][odat * ndim + idim] = *in_data;

            if (output_time_total)
            {
              if (input_voltage)
                time_total[ipol][odat] += val * val;
              else
                time_total[ipol][odat] += val;
            }
            in_data++;
          }
        }
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

void dsp::RescaleMedianMadCalculator::compute()
{
  auto nval = ndim * nintegrated;
  // first find the median
  for (unsigned ipol = 0; ipol < npol; ipol++)
  {
    for (unsigned ichan = 0; ichan < nchan; ichan++)
    {
      if (dsp::Operation::verbose)
        cerr << "dsp::RescaleMedianMadCalculator::compute finding median for "
          << " ipol=" << ipol
          << ", ichan=" << ichan
          << endl;

      float *data = &(_sample_data[ipol][ichan][0]);
      auto median = find_median(data, nval);

      // update sample data to now be absolute deviation from median
      for (uint64_t ival = 0; ival < nval; ival++)
        absolute_deviation[ipol][ichan][ival] = fabs(data[ival] - median);

      auto mad = find_median((&absolute_deviation[ipol][ichan][0]), nval);
      auto std_est = mad / scale_factor;

      if (dsp::Operation::verbose)
        cerr << "dsp::RescaleMedianMadCalculator::compute "
          << "Median[" << ipol << "][" << ichan << "] = " << median
          << ", MAD[" << ipol << "][" << ichan << "] = " << mad
          << ", std=" << std_est
          << endl;

      offset[ipol][ichan] = -median;
      mean[ipol][ichan] = static_cast<double>(median);
      variance[ipol][ichan] = static_cast<double>(std_est * std_est);
      scale[ipol][ichan] = (std_est > 0) ? 1.0 / std_est : 1.0;
    }
  }
}

void dsp::RescaleMedianMadCalculator::reset_sample_data()
{
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleMedianMadCalculator::reset_sample_data" << endl;

  nintegrated = 0;
  for (unsigned ipol=0; ipol < npol; ipol++)
  {
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      zero (_sample_data[ipol][ichan]);
    }

    if (ipol < time_total.size())
    {
      zero (time_total[ipol]);
    }
  }
  if (dsp::Operation::verbose)
    cerr << "dsp::RescaleMedianMadCalculator::reset_sample_data - complete" << endl;
}

const double* dsp::RescaleMedianMadCalculator::get_mean (unsigned ipol) const
{
  assert (ipol < mean.size());
  return &(mean[ipol][0]);
}

const double* dsp::RescaleMedianMadCalculator::get_variance (unsigned ipol) const
{
  assert (ipol < variance.size());
  return &(variance[ipol][0]);
}

void dsp::RescaleMedianMadCalculator::set_scale_factor (float _scale_factor)
{
  if (_scale_factor == 0.0)
    throw Error (InvalidParam, "dsp::RescaleMedianMadCalculator::set_scale_factor",
		 "invalid scale_factor=", _scale_factor);

  scale_factor = _scale_factor;
}

inline void swap(float *data, uint64_t i, uint64_t j)
{
  if (i == j)
    return;

  auto tmp_val = data[i];
  data[i] = data[j];
  data[j] = tmp_val;
}

float dsp::RescaleMedianMadCalculator::find_median(float *data, uint64_t nsample)
{
  // the median of medians algorithm pseudo code assumes 1 offset but C++ uses zero offset
  auto n = (nsample - 1) / 2 ;
  auto idx = select(data, 0, nsample - 1, n);
  assert (idx == n);
  return data[idx];
}

uint64_t dsp::RescaleMedianMadCalculator::select(float *data, uint64_t left, uint64_t right, uint64_t n)
{
  assert (left <= right);
  assert (left <= n);
  assert (n <= right);

  // left and right are inclusive ranges
  while (left < right)
  {
    // find location of median of medians
    auto pivot_idx = pivot(data, left, right);

    // partition data around median-of-medians value and
    // find bound of where where n is in relation to partition
    auto partition_idx = partition(data, left, right, pivot_idx, n);

    // partition group has median value
    if (n == partition_idx)
      return n;

    if (n < partition_idx)
    {
      // n-th value was in group less than median of medians update right bound
      right = partition_idx - 1;
    }
    else
    {
      // n-th value was in group greater than median of medians update left bound
      left = partition_idx + 1;
    }
  }
  // left == right
  return left;
}

uint64_t dsp::RescaleMedianMadCalculator::pivot(float *data, uint64_t left, uint64_t right)
{
  assert (left <= right);
  // left and right are inclusive ranges
  if ((right - left) < 5) {
    return partition5(data, left, right);
  }

  // split data into chunks of upto 5 and find median, last group may have a length of less than 5
  for (auto idx = left; idx <= right; idx += 5)
  {
    // handle last chunk may have 5 or fewer elements
    auto sub_right = idx + 4;
    if (sub_right > right)
      sub_right = right;

    auto median5 = partition5(data, idx, sub_right);
    auto swap_idx = left + (idx - left) / 5;
    swap(data, median5, swap_idx);
  }

  auto mid = (right - left) / 10 + left + 1;
  return select(data, left, left + (right - left) / 5, mid);
}

uint64_t dsp::RescaleMedianMadCalculator::partition(float *data, uint64_t left, uint64_t right, uint64_t pivot_idx, uint64_t n)
{
  assert (left <= right);

  auto pivot_val = data[pivot_idx];
  // move pivot value to the right, easier to handle
  swap(data, pivot_idx, right);

  // move all values less than pivot value to the left (i.e. group 1)
  auto store_idx = left;
  for (auto i = left; i < right; i++)
  {
    if (data[i] < pivot_val)
    {
      swap(data, i, store_idx);
      store_idx++;
    }
  }

  // move all values with the same as pivot value into group 2
  auto store_idx_eq = store_idx;
  for (auto i = store_idx; i < right; i++)
  {
    if (data[i] == pivot_val)
    {
      swap(data, i, store_idx_eq);
      store_idx_eq++;
    }
  }

  // after this, all values beyond store_idx_eq are greater than pivot value
  // and are in group 3.
  swap(data, right, store_idx_eq);

  // n-th smallest value is less than the pivot value, return upper bound of group 1
  if (n < store_idx)
    return store_idx;

  // n-th smallest is in group with the pivot value, return n as the index
  if (n <= store_idx_eq)
    return n;

  // n-th smallest value is greater than the pivot value, return lower bound of group 3
  return store_idx_eq;
}

uint64_t dsp::RescaleMedianMadCalculator::partition5(float *data, uint64_t left, uint64_t right)
{
  assert (left <= right);

  auto i = left + 1;
  while (i <= right)
  {
    auto j = i;
    while (j > left && data[j - 1] > data[j])
    {
      swap(data, j - 1, j);
      j--;
    }
    i++;
  }

  return left + (right - left) / 2;
}
