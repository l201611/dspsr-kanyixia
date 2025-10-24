//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019-2025 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/UWBTwoBitUnpacker.h"
#include "dsp/ASCIIObservation.h"

#include "Error.h"

#include <errno.h>
#include <string.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::UWBTwoBitUnpacker::UWBTwoBitUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::UWBTwoBitUnpacker ctor" << endl;

  set_ndig (2);
  set_nstate (4);

  npol = 2;
  ndim = 2;

  have_scales_and_offsets = false;
}

dsp::UWBTwoBitUnpacker::~UWBTwoBitUnpacker ()
{
}

unsigned dsp::UWBTwoBitUnpacker::get_output_offset (unsigned idig) const
{
  return idig % 2;
}

unsigned dsp::UWBTwoBitUnpacker::get_output_ipol (unsigned idig) const
{
  return idig / 2;
}

unsigned dsp::UWBTwoBitUnpacker::get_output_ichan (unsigned idig) const
{
  return idig / 4;
}

unsigned dsp::UWBTwoBitUnpacker::get_ndim_per_digitizer () const
{
  return 1;
}

dsp::UWBTwoBitUnpacker * dsp::UWBTwoBitUnpacker::clone () const
{
  return new UWBTwoBitUnpacker (*this);
}

//! Return true if the unpacker support the specified output order
bool dsp::UWBTwoBitUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return (order == TimeSeries::OrderFPT);
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::UWBTwoBitUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

bool dsp::UWBTwoBitUnpacker::matches (const Observation* observation)
{
  if (!dynamic_cast<const ASCIIObservation *>(observation))
  {
    if (verbose)
      cerr << "dsp::UWBTwoBitUnpacker::matches"
              " ASCIIObservation required and not available" << endl;
    return false;
  }

  return (observation->get_machine()== "UWB" || observation->get_machine()== "Medusa" || observation->get_machine()== "Apollo")
    && observation->get_nchan() == 1
    && observation->get_ndim() == 2
    && (observation->get_npol() == 2 || observation->get_npol() == 1)
    && observation->get_nbit() == 2;
}

void dsp::UWBTwoBitUnpacker::get_scales_and_offsets ()
{
  unsigned nchan = input->get_nchan();
  unsigned npol = input->get_npol();

  const Input * in = input->get_loader();
  const Observation * obs = in->get_info();
  const ASCIIObservation * info = dynamic_cast<const ASCIIObservation *>(obs);
  if (!info)
    throw Error (InvalidState, "dsp::UWBTwoBitUnpacker::get_scales_and_offsets",
                 "ASCIIObservation required and not available");

  if (verbose)
    cerr << "dsp::UWBTwoBitUnpacker::get_scales_and_offsets nchan=" << nchan << " npol=" << npol << endl;

  scales.resize(nchan);
  offsets.resize(nchan);
  stringstream key;
  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    offsets[ichan].resize(npol);
    scales[ichan].resize(npol);
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      key.str("");
      key << "DAT_SCL_" << ichan << "_" << ipol;
      try
      {
        info->custom_header_get (key.str().c_str(), "%f", &(scales[ichan][ipol]));
      }
      catch (Error& error)
      {
        scales[ichan][ipol] = 1.0f;
      }

      key.str("");
      key << "DAT_OFF_" << ichan << "_" << ipol;
      try
      {
        info->custom_header_get (key.str().c_str(), "%f", &(offsets[ichan][ipol]));
      }
      catch (Error& error)
      {
        offsets[ichan][ipol] = 0;
      }
      if (verbose)
      {
        cerr << "scales[" << ichan << "][" << ipol << "]=" << scales[ichan][ipol] << endl;
        cerr << "offsets[" << ichan << "][" << ipol << "]=" << offsets[ichan][ipol] << endl;
      }
    }
  }
}

void dsp::UWBTwoBitUnpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::UWBTwoBitUnpacker::unpack()" << endl;

  npol = input->get_npol();
  set_ndig (npol*2);

  if (!have_scales_and_offsets)
  {
    if (verbose)
      cerr << "dsp::UWBTwoBitUnpacker::unpack getting scales and offsets" << endl;
    get_scales_and_offsets();
    have_scales_and_offsets = true;
  }


  // Data are stored in TFP order, but nchan == 1, so TP order
  unsigned ichan = 0;
  unsigned long * hists[2];

  const uint64_t ndat = input->get_ndat();
  const unsigned into_stride = ndim;
  const unsigned from_stride = 1;

  if (verbose)
    cerr << "dsp::UWBTwoBitUnpacker::unpack ndat=" << ndat
         << " ndim=" << ndim << " npol=" << npol << endl;

  char * from = ((char *) input->get_rawptr()) + 0;
  float * into_p0 = output->get_datptr (ichan, 0);
  float * into_p1 = output->get_datptr (ichan, 1);
  hists[0] = get_histogram (0);
  hists[1] = get_histogram (1);
  hists[2] = get_histogram (2);
  hists[3] = get_histogram (3);

  //const float offset = offsets[ichan][ipol];
  //const float scale = scales[ichan][ipol];
  const float offset = 0;
  const float scale = 1;

  for (unsigned idat=0; idat<ndat; idat++)
  {
    char packed = (char) (*from);

    int8_t real_p0 = int8_t((packed & 0x03) << 6) / 64;
    int8_t imag_p0 = int8_t((packed & 0x0c) << 4) / 64;
    int8_t real_p1 = int8_t((packed & 0x30) << 2) / 64;
    int8_t imag_p1 = int8_t(packed & 0xc0)        / 64;

    into_p0[0] = (float(real_p0) * scale) + offset;
    into_p0[1] = (float(imag_p0) * scale) + offset;
    into_p0[0] = (float(real_p0) * scale) + offset;
    into_p1[1] = (float(imag_p0) * scale) + offset;

    real_p0 = max(int8_t(-2), real_p0);
    real_p0 = min(int8_t(1), real_p0);
    imag_p0 = max(int8_t(-2), imag_p0);
    imag_p0 = min(int8_t(1), imag_p0);
    real_p1 = max(int8_t(-2), real_p1);
    real_p1 = min(int8_t(1), real_p1);
    imag_p1 = max(int8_t(-2), imag_p1);
    imag_p1 = min(int8_t(1), imag_p1);

    hists[0][real_p0+2]++;
    hists[1][imag_p0+2]++;
    hists[2][real_p1+2]++;
    hists[3][imag_p1+2]++;

    into_p0 += into_stride;
    into_p1 += into_stride;
    from += from_stride;

  } //each sample
}

