/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/KnownDataSource.h"
#include "dsp/OperationPerformanceMetrics.h"
#include "dsp/WeightedTimeSeries.h"

#include <cstring>

void dsp::test::KnownDataSource::operation()
{
  const auto ndat = output->get_ndat();
  const auto nchan = output->get_nchan();
  const auto npol = output->get_npol();
  const auto ndim = output->get_ndim();
  const auto offset = ndat * nchan * ndim * npol * iterations;

  performance_metrics->update_metrics(output);

  if (verbose)
    std::cerr << "dsp::test::KnownDataSource::operation() - ndat=" << ndat
      << ", nchan=" << nchan << ", ndim=" << ndim << ", npol=" << npol
      << ", iterations=" << iterations << ", output->get_rate()=" << output->get_rate() << std::endl;

  switch (output_order)
  {
  case dsp::TimeSeries::OrderFPT:
    {
      uint64_t nval = ndat * ndim;
      for (auto ichan = 0; ichan < nchan; ichan++)
      {
        for (auto ipol = 0; ipol < npol; ipol++)
        {
          float *ptr = output->get_datptr(ichan, ipol);
          auto chanpol_offset = offset + ((ichan * npol) + ipol) * nval;
          memcpy(ptr, &data[chanpol_offset], nval * sizeof(float));
        }
      }
    }

    break;

  case dsp::TimeSeries::OrderTFP:
  default:
    {
      float *ptr = output->get_dattfp();
      uint64_t nval = ndat * nchan * npol * ndim;
      memcpy(ptr, data.data() + offset, nval * sizeof(float));
    }

    break;
  }

  auto weighted_output = dynamic_cast<WeightedTimeSeries*>(output);
  if (weighted_output)
  {
    const auto nchan_weight = weighted_output->get_nchan_weight();
    const auto npol_weight = weighted_output->get_npol_weight();

    if (verbose)
      std::cerr << "dsp::test::KnownDataSource::operation() - using weighted time series" << std::endl;

    auto nweight = weighted_output->get_nweights();
    if (verbose)
      std::cerr << "dsp::test::KnownDataSource::operation() - nweight=" << nweight << std::endl;

    uint64_t idx = iterations * nchan_weight * npol_weight * nweight;
    if (verbose)
      std::cerr << "dsp::test::KnownDataSource::operation() - idx=" << idx << std::endl;

    // WTS is always FPT and we can use the ndat
    for (auto ichan = 0; ichan < nchan_weight; ichan++)
    {
      for (auto ipol = 0; ipol < npol_weight; ipol++)
      {
        auto weights_ptr = weighted_output->get_weights(ichan, ipol);
        for (auto iweight = 0; iweight < nweight; iweight++, idx++)
        {
          weights_ptr[iweight] = weights[idx];
        }
      }
    }
  }

  output->set_input_sample(current_samples);

  current_samples += ndat;
  iterations++;

  set_end_of_data(iterations >= niterations);

  if (verbose)
    std::cerr << "dsp::test::KnownDataSource::operation() - complete. eod=" << eod << std::endl;

}

dsp::Source* dsp::test::KnownDataSource::clone() const
{
  dsp::test::KnownDataSource * clone = new dsp::test::KnownDataSource();
  clone->set_data(data);
  clone->set_weights(weights);
  clone->set_output(output);
  return clone;
}


