/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ChanPolSelect.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/TFPOffset.h"

#include <assert.h>
#include <cstring>

using namespace std;

dsp::ChanPolSelect::ChanPolSelect ()
  : Transformation<TimeSeries,TimeSeries> ("ChanPolSelect", anyplace)
{
}

void dsp::ChanPolSelect::set_engine(Engine* _engine)
{
  if (verbose)
  {
    cerr << "dsp::ChanPolSelect::set_engine" << endl;
  }

  engine = _engine;
}

void dsp::ChanPolSelect::set_start_channel_index(unsigned channel_index)
{
  start_channel_index = channel_index;
  if (verbose)
  {
    cerr << "dsp::ChanPolSelect::set_start_channel_index"
    << " start_channel_index=" << start_channel_index
    << " channel_index=" << channel_index
    << endl;
  }
}

void dsp::ChanPolSelect::set_number_of_channels_to_keep(unsigned nchan)
{
  number_of_channels_to_keep = nchan;
  if (verbose)
  {
    cerr << "dsp::ChanPolSelect::set_number_of_channels_to_keep"
    << " number_of_channels_to_keep=" << number_of_channels_to_keep
    << " nchan=" << nchan
    << endl;
  }
}

void dsp::ChanPolSelect::set_start_polarization_index(unsigned polarization_index)
{
  start_polarization_index = polarization_index;
  if (verbose)
  {
    cerr << "dsp::ChanPolSelect::set_start_polarization_index"
    << " start_polarization_index=" << start_polarization_index
    << " polarization_index=" << polarization_index
    << endl;
  }
}

void dsp::ChanPolSelect::set_number_of_polarizations_to_keep(unsigned npol)
{
  number_of_polarizations_to_keep = npol;
  if (verbose)
  {
    cerr << "dsp::ChanPolSelect::set_number_of_polarizations_to_keep"
    << " number_of_polarizations_to_keep=" << number_of_polarizations_to_keep
    << " npol=" << npol
    << endl;
  }
}

unsigned dsp::ChanPolSelect::get_start_channel_index() const
{
  return start_channel_index;
}

unsigned dsp::ChanPolSelect::get_number_of_channels_to_keep() const
{
  return number_of_channels_to_keep;
}

unsigned dsp::ChanPolSelect::get_start_polarization_index() const
{
  return start_polarization_index;
}

unsigned dsp::ChanPolSelect::get_number_of_polarizations_to_keep() const
{
  return number_of_polarizations_to_keep;
}

void dsp::ChanPolSelect::prepare ()
{
  if (prepared)
  {
    return;
  }

  if (engine)
  {
    if (verbose)
    {
      cerr << "dsp::ChanPolSelect::prepare call Engine::setup" << endl;
    }
    engine->setup (this);
  }

  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol  = input->get_npol();

  if (verbose)
  {
    cerr << "dsp::ChanPolSelect::prepare"
      << " nchan=" << input_nchan
      << " npol=" << input_npol
      << " start_channel_index=" << start_channel_index
      << " number_of_channels_to_keep=" << number_of_channels_to_keep
      << " start_polarization_index=" << start_polarization_index
      << " number_of_polarizations_to_keep=" << number_of_polarizations_to_keep
      << endl;
  }

  // Confirm Validity of input parameters

  if (number_of_channels_to_keep == 0)
  {
    throw Error (InvalidParam, "dsp::ChanPolSelect::prepare()",
      "number_of_channels_to_keep=%u == 0",
      number_of_channels_to_keep);
  }

  if (number_of_polarizations_to_keep == 0)
  {
    throw Error (InvalidParam, "dsp::ChanPolSelect::prepare()",
      "number_of_polarizations_to_keep=%u == 0",
      number_of_polarizations_to_keep);
  }

  if (start_channel_index + number_of_channels_to_keep > input_nchan)
  {
    throw Error (InvalidParam, "dsp::ChanPolSelect::prepare()",
      "start_channel_index=%u + number_of_channels_to_keep=%u > input_nchan=%u",
      start_channel_index, number_of_channels_to_keep, input_nchan);
  }

  if (start_polarization_index + number_of_polarizations_to_keep > input_npol)
  {
    throw Error (InvalidParam, "dsp::ChanPolSelect::prepare()",
      "start_polarization_index=%u + number_of_polarizations_to_keep=%u > input_npol=%u",
      start_polarization_index, number_of_polarizations_to_keep, input_npol);
  }

  output->copy_configuration(input);
  output->set_nchan(number_of_channels_to_keep);
  output->set_npol(number_of_polarizations_to_keep);

  auto weighted_output = dynamic_cast<WeightedTimeSeries*>(output.get());
  auto weighted_input = dynamic_cast<const WeightedTimeSeries*>(input.get());

  // copy weights from input to output

  if (weighted_output && weighted_input && weighted_input->get_ndat_per_weight() > 0)
  {
    if (verbose)
      std::cerr << "dsp::ChanPolSelect::prepare input and output WeightedTimeSeries" << std::endl;

    uint64_t nchan_weight = weighted_input->get_nchan_weight();
    uint64_t npol_weight = weighted_input->get_npol_weight();

    wt_start_channel_index = 0;
    wt_number_of_channels_to_keep = 1;
    wt_start_polarization_index = 0;
    wt_number_of_polarizations_to_keep = 1;

    if (nchan_weight == 1)
    {
      if (verbose)
        std::cerr << "dsp::ChanPolSelect::prepare nchan_weight==1" << std::endl;
    }
    else if (nchan_weight==input_nchan)
    {
      if (verbose)
        std::cerr << "dsp::ChanPolSelect::prepare nchan_weight==input_nchan" << std::endl;
      wt_start_channel_index = start_channel_index;
      wt_number_of_channels_to_keep = number_of_channels_to_keep;
    }
    else
    {
      throw Error(InvalidParam, "dsp::ChanPolSelect::prepare", "implemented only for nchan_weight==1 or nchan_weight==nchan");
    }

    if (npol_weight == 1)
    {
      if (verbose)
        std::cerr << "dsp::ChanPolSelect::prepare npol_weight==1" << std::endl;
    }
    else if (npol_weight==input_npol)
    {
      if (verbose)
        std::cerr << "dsp::ChanPolSelect::prepare npol_weight==input_npol" << std::endl;
      wt_start_polarization_index = start_polarization_index;
      wt_number_of_polarizations_to_keep = number_of_polarizations_to_keep;
    }
    else
    {
      throw Error(InvalidParam, "dsp::ChanPolSelect::prepare", "implemented only for npol_weight==1 or npol_weight==npol");
    }

    if (verbose)
      std::cerr << "dsp::ChanPolSelect::prepare npol_weight=" << wt_number_of_polarizations_to_keep << " nchan_weight=" << wt_number_of_channels_to_keep << std::endl;
    weighted_output->set_nchan_weight(wt_number_of_channels_to_keep);
    weighted_output->set_npol_weight(wt_number_of_polarizations_to_keep);
  }

  prepared = true;
}

void dsp::ChanPolSelect::transformation ()
{
  if (verbose)
    cerr << "dsp::ChanPolSelect::transformation" << endl;

  prepare();

  const uint64_t input_ndat = input->get_ndat();
  const unsigned input_ndim = input->get_ndim();

  // copy the transient attributes of the input (avoids resetting attributes set once in prepare)
  output->copy_transient_attributes(input);
  output->resize(input_ndat);

  if (input_ndat == 0)
  {
    return;
  }

  if (engine)
  {
    if (verbose)
    {
      cerr << "dsp::ChanPolSelect::transformation calling Engine::select" << endl;
    }
    engine->select(input,output);
    return;
  }

  auto weighted_output = dynamic_cast<WeightedTimeSeries*>(output.get());
  auto weighted_input = dynamic_cast<const WeightedTimeSeries*>(input.get());

  // copy weights from input to output

  if (weighted_output && weighted_input && weighted_input->get_ndat_per_weight() > 0)
  {
    if (verbose)
      std::cerr << "dsp::ChanPolSelect::transformation copying weights into WeightedTimeSeries" << std::endl;

    uint64_t nweights = weighted_input->get_nweights();
    const uint64_t weights_size = sizeof(uint16_t) * nweights;

    for (unsigned ipol=0; ipol < wt_number_of_polarizations_to_keep; ipol++)
    {
      for (unsigned ichan=0; ichan < wt_number_of_channels_to_keep; ichan++)
      {
        memcpy(
          weighted_output->get_weights(ichan, ipol),
          weighted_input->get_weights(ichan+wt_start_channel_index, ipol+wt_start_polarization_index),
          weights_size);
      }
    }
  }

  switch (input->get_order())
  {
    case dsp::TimeSeries::OrderTFP:
    {
      if (verbose)
      {
        cerr << "dsp::ChanPolSelect::transformation OrderTFP" << endl;
      }

      const float* in_data = input->get_dattfp();
	    float* out_data = output->get_dattfp();

      TFPOffset input_offset(input);
      TFPOffset output_offset(output);

      in_data += input_offset(0, start_channel_index, start_polarization_index);

      for (uint64_t idat=0; idat < input_ndat; idat++)
      {
        for (unsigned ichan=0; ichan < number_of_channels_to_keep; ichan++)
        {
          for (unsigned ipol=0; ipol < number_of_polarizations_to_keep; ipol++)
          {
            auto input_index = input_offset(idat, ichan, ipol);
            auto output_index = output_offset(idat, ichan, ipol);

            for (unsigned idim=0; idim < input_ndim; idim++)
            {
              out_data[output_index + idim] = in_data[input_index + idim];
            }
          }
        }
      }
    }
    break;

    case dsp::TimeSeries::OrderFPT:
    {
      if (verbose)
      {
        cerr << "dsp::ChanPolSelect::transformation OrderFPT" << endl;
      }

      const uint64_t data_size = sizeof(float) * input_ndat * input_ndim;

      for (unsigned ipol=0; ipol < number_of_polarizations_to_keep; ipol++)
      {
        for (unsigned ichan=0; ichan < number_of_channels_to_keep; ichan++)
        {
          memcpy(
            output->get_datptr(ichan, ipol),
            input->get_datptr(ichan+start_channel_index, ipol+start_polarization_index),
            data_size);
        }
      }
    }
    break;
  }
}

void dsp::ChanPolSelect::Engine::setup (ChanPolSelect* user)
{
  start_channel_index = user->get_start_channel_index();
  number_of_channels_to_keep = user->get_number_of_channels_to_keep();
  start_polarization_index = user->get_start_polarization_index();
  number_of_polarizations_to_keep = user->get_number_of_polarizations_to_keep();
}
