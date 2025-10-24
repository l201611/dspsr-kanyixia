/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "BoxMuller.h"
#include <random>
#include <algorithm>

#include <dsp/GaussianNoiseSource.h>

dsp::test::GaussianNoiseSource::GaussianNoiseSource(unsigned _niterations) : TestSource("GaussianNoiseSource", _niterations) {}

void dsp::test::GaussianNoiseSource::operation()
{
  const auto ndat = output->get_ndat();
  const auto nchan = output->get_nchan();
  const auto ndim = output->get_ndim();
  const auto npol = output->get_npol();

  if (verbose)
    std::cerr << "GaussianNoiseSource::operation() - ndat=" << ndat
      << ", nchan=" << nchan << ", ndim=" << ndim << ", npol=" << npol
      << ", iterations=" << iterations << ", output->get_rate()=" << output->get_rate() << std::endl;

  time_t now = time(nullptr);
  BoxMuller bm(now);
  switch (output_order)
  {
  case dsp::TimeSeries::OrderFPT:
    {
      uint64_t nval = ndat * ndim;
      std::vector<float> data(nval, 0.0);
      for (auto ichan = 0; ichan < nchan; ichan++)
      {
        for (auto ipol = 0; ipol < npol; ipol++)
        {
          std::generate(data.begin(), data.end(), bm);
          float *ptr = output->get_datptr(ichan, ipol);

          for (auto ival = 0; ival < nval; ival++)
          {
            ptr[ival] = data[ival];
          }
        }
      }
    }

    break;

  case dsp::TimeSeries::OrderTFP:
  default:
    {
      float *ptr = output->get_dattfp();
      uint64_t nval = ndat * nchan * npol * ndim;
      std::vector<float> data(nval, 0.0);
      std::generate(data.begin(), data.end(), bm);
      for (auto ival = 0; ival < nval; ival++)
      {
        ptr[ival] = data[ival];
      }
    }

    break;
  }

  iterations++;
  set_end_of_data(iterations >= niterations);

  if (verbose)
    std::cerr << "GaussianNoiseSource::operation() - complete. eod=" << eod << std::endl;
}

dsp::Source* dsp::test::GaussianNoiseSource::clone() const
{
  GaussianNoiseSource * clone = new dsp::test::GaussianNoiseSource();
  clone->set_output(output);
  return clone;
}
