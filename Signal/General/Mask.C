/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Mask.h"

#include <assert.h>

using namespace std;

dsp::Mask::Mask(const char* name) 
  : Transformation<TimeSeries,TimeSeries>(name,inplace)
{
  if (verbose)
    cerr << "dsp::Mask::Mask()" << endl;
}

dsp::Mask::~Mask ()
{
  if (verbose)
    cerr << "dsp::Mask::~Mask()" << endl;
}

void dsp::Mask::transformation()
{
  if (verbose)
    cerr << "dsp::Mask::transformation" << endl;
  
  mask_data();

  // indicate the output timeseries contains zeroed data
  output->set_zeroed_data (true);
}

bool dsp::Mask::get_order_supported(TimeSeries::Order order) const
{
  if (order == TimeSeries::OrderFPT || order == TimeSeries::OrderTFP)
    return true;
  return false;
}

void dsp::Mask::mask (uint64_t start_idat, uint64_t end_idat)
{
  if (verbose)
    cerr << "dsp::Mask::mask start=" << start_idat << " end=" << end_idat << endl;

  switch (output->get_order())
  {
    case dsp::TimeSeries::OrderTFP:
      mask_TFP(start_idat, end_idat);
      break;
    case dsp::TimeSeries::OrderFPT:
      mask_FPT(start_idat, end_idat);
      break;
    default:
    {
      throw Error (InvalidState, "dsp::Mask::mask", "unsupported order");
    }
  }
}

void dsp::Mask::mask_TFP(uint64_t start_idat, uint64_t end_idat)
{
  if (verbose)
    cerr << "dsp::Mask::mask_TFP: OrderTFP" << endl;

  assert (end_idat <= output->get_ndat());

  const unsigned nchan = output->get_nchan();
  const unsigned npol = output->get_npol();
  const unsigned ndim = output->get_ndim();
  
  const uint64_t stride = nchan * npol * ndim;

  for (uint64_t idat=start_idat; idat < end_idat; idat++)
  {
    float* dat = output->get_dattfp() + idat * stride;
    for (unsigned ifloat=0; ifloat < stride; ifloat++)
      dat[ifloat] = 0.0;
  }
}

void dsp::Mask::mask_FPT(uint64_t start_idat, uint64_t end_idat)
{
  if (verbose)
    cerr << "dsp::Mask::mask_FPT: OrderFPT" << endl;

  assert (end_idat <= output->get_ndat());

  const unsigned nchan = output->get_nchan();
  const unsigned npol = output->get_npol();
  const unsigned ndim = output->get_ndim();

  const uint64_t start_ifloat = start_idat * ndim;
  const uint64_t end_ifloat = end_idat * ndim;

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      float* dat = output->get_datptr (ichan, ipol);
      for (uint64_t ifloat=start_ifloat; ifloat < end_ifloat; ifloat++)
        dat[ifloat] = 0.0;      
    }
  }
}


#include "dsp/MaskTimes.h"

// lazy factory for now - apologies to future developer
dsp::Mask* dsp::Mask::factory (const std::string& descriptor)
{
  Reference::To<MaskTimes> times = new MaskTimes;
  times->load(descriptor);
  return times.release();
}

