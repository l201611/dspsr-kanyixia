/***************************************************************************
 *
 *   Copyright (C) 2008-2025 by Willem van Straten, Will Gauvin and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Rescale.h"
#include "dsp/RescaleMeanStdCalculator.h"
#include "dsp/InputBuffering.h"

#include <assert.h>

using namespace std;

dsp::Rescale::Rescale ()
  : Transformation<TimeSeries,TimeSeries> ("Rescale", anyplace)
{
  // default calculator is the Mean and standard deviation
  calculator = new dsp::RescaleMeanStdCalculator;
}

void dsp::Rescale::set_output_after_interval (bool flag)
{
  output_after_interval = flag;
}

void dsp::Rescale::set_output_time_total (bool flag)
{
  output_time_total = flag;
}

void dsp::Rescale::set_constant (bool value)
{
  constant_offset_scale = value;
}

void dsp::Rescale::set_decay (float _decay_constant)
{
	if (_decay_constant > 0)
		do_decay=true;
	decay_constant=_decay_constant;
}

void dsp::Rescale::set_interval_seconds (double seconds)
{
  interval_seconds = seconds;
}

void dsp::Rescale::set_interval_samples (uint64_t samples)
{
  interval_samples = samples;
}

void dsp::Rescale::set_exact (bool value)
{
  exact = value;
  if (exact && !interval_samples)
    throw Error(InvalidState, "dsp::Rescale::set_exact",
        "interval_sample == 0 (must be set)");

  // only apply the buffering policy if exact is true
  if (!exact)
    return;

  if (!has_buffering_policy())
    set_buffering_policy( new InputBuffering (this) );

  get_buffering_policy()->set_maximum_samples (interval_samples);
}

template<typename T>
void zero (vector<T>& data)
{
  const unsigned n = data.size();
  for (unsigned i=0; i<n; i++)
    data[i]=0;
}

void dsp::Rescale::init ()
{
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_ndat = input->get_ndat();

  if (verbose)
    cerr << "dsp::Rescale::init npol=" << input_npol
         << " nchan=" << input_nchan
         << " ndat=" << input_ndat << endl;

  if (interval_samples)
    nsample = interval_samples;
  else if (interval_seconds)
    nsample = uint64_t( interval_seconds * input->get_rate() );
  else
  {
    if (input_ndat)
    {
      nsample = input_ndat;
    }
    else
    {
      if (verbose)
        cerr << "dsp::Rescale::init ndat=0, exiting init as there is no data" << endl;
      return;
    }
  }

  if (verbose)
    cerr << "dsp::Rescale::init interval samples = " << nsample << endl;

  if (!nsample)
  {
    Error error (InvalidState, "dsp::Rescale::init", "nsample == 0");
    error << " (interval samples=" << interval_samples
	  << " seconds=" << interval_seconds << " ndat=" << input_ndat << ")";
    throw error;
  }

  if (engine)
  {
    engine->init(input, nsample, exact, constant_offset_scale);
    return;
  }

  if (!calculator)
    calculator = new RescaleMeanStdCalculator;

  calculator->init(input, nsample, output_time_total);

  isample = 0;
  first_integration = true;

  if (do_decay)
  {
	  decay_offset.resize (input_npol);

    for (unsigned ipol=0; ipol < input_npol; ipol++)
    {
      decay_offset[ipol].resize(input_nchan);
      zero (decay_offset[ipol]);
    }
  }
}

void dsp::Rescale::transformation ()
{
  if (verbose)
    cerr << "dsp::Rescale::transformation" << endl;

  // if requested a minimum number of samples, let input buffering handle it
  if (exact && (input->get_ndat() < interval_samples))
  {
    if (verbose)
      cerr << "dsp::Rescale::transformation waiting for additional samples" << endl;
    get_buffering_policy()->set_next_start ( 0 );
    output->set_ndat (0);
    return;
  }

  if (verbose)
    cerr << "dsp::Rescale::transformation input->get_input_sample()="
      << input->get_input_sample()
      << endl;

  if (!nsample)
    init();

  const uint64_t input_ndat = exact ? interval_samples : input->get_ndat();
  const uint64_t output_ndat = input->get_ndat();

  if (verbose)
    cerr << "dsp::Rescale::transformation input_ndat=" << input_ndat
         << " output_ndat=" << output_ndat
         << " nsample=" << nsample << endl;

  // prepare the output TimeSeries
  output->copy_configuration (input);

  // Since we will be rescaling data, remove any pre-set scale
  // factors (for example Filterbank/FFT normalizations).
  output->set_scale(1.0);

  if (output != input)
    output->resize (output_ndat);
  else
    output->set_ndat (output_ndat);

  if (!output_ndat)
    return;

  if (engine)
  {
    if (verbose)
      cerr << "dsp::Rescale::transformation using supplied engine" << endl;

    engine->transform(input, output);

    if (exact)
      get_buffering_policy()->set_next_start(interval_samples);

    if (verbose)
      cerr << "dsp::Rescale::transformation engine->transform() has exited" << endl;

    return;
  }

  if (verbose)
    cerr << "dsp::Rescale::transformation using default on CPU implementation" << endl;

  // only perform the rescale if we're constant and have done the first integration
  if (constant_offset_scale && !first_integration)
  {
    rescale(0, input_ndat);

    // ensure we update the buffering policy if using exact number of samples
    if (exact)
      get_buffering_policy()->set_next_start ( interval_samples );

    return;
  }

  uint64_t start_dat = 0;
  uint64_t end_dat = 0;

  while (start_dat < input_ndat)
  {
    // default to sampling until the end
    auto nsamp = input_ndat - start_dat;

    // don't sample more than needed to calculate stats
    if (isample + nsamp > nsample)
      nsamp = nsample - isample;

    end_dat = start_dat + nsamp;

    if (dsp::Operation::verbose)
      cerr << "dsp::Rescale::transformation "
        << "start_dat=" << start_dat
        << ", end_dat=" << end_dat
        << ", nsamp=" << nsamp
        << ", isample=" << isample
        << endl;

    if (first_integration || !constant_offset_scale)
      isample = calculator->sample_data(input, start_dat, end_dat, output_time_total);

    // perform calculation of scales and offsets if:
    // * we have sampled enough time samples to perform a calculation
    // * or, we are at the end of the input data on the first integration
    if ((isample == nsample) || (first_integration && end_dat == input_ndat))
    {
      calculator->compute();
      if (verbose)
        cerr << "dsp::Rescale::transformation scales and offset updated" << endl;

      // ensure we have the correct time when scales and offsets were updated
      if (update_epoch == MJD::zero)
        update_epoch = input->get_start_time();
      update_epoch += isample / input->get_rate();

      fire_scales_updated(input, start_dat);

      // reset calculator and isample if we have integrated the exact amount.
      if (isample == nsample)
      {
        first_integration = false;
        isample = 0;
        calculator->reset_sample_data();
      }
    }

    // this will allow breaking out of loop but also to ensure we
    // apply rescaling to the rest of the input data.
    if (constant_offset_scale && !first_integration)
      end_dat = input_ndat;

    rescale(start_dat, end_dat);
    start_dat = end_dat;
  }

  if (dsp::Operation::verbose)
    cerr << "dsp::Rescale::transformation exiting" << endl;
}

void dsp::Rescale::rescale(uint64_t start_dat, uint64_t end_dat)
{
  const unsigned input_ndim  = input->get_ndim();
  const unsigned input_npol  = input->get_npol();
  const unsigned input_nchan = input->get_nchan();

  const auto& offsets = calculator->get_offsets();
  const auto& scales = calculator->get_scales();

  //  Apply scale and offsets
  switch (input->get_order())
  {
    case TimeSeries::OrderTFP:
    {
      float tmp;
      const float* in_data = input->get_dattfp();
      float* out_data = output->get_dattfp();
      in_data += start_dat * input_nchan * input_npol * input_ndim;
      out_data += start_dat * input_nchan * input_npol * input_ndim;
      for (unsigned idat=start_dat; idat < end_dat; idat++)
      {
        for (unsigned ichan=0; ichan < input_nchan; ichan++)
        {
          for (unsigned ipol=0; ipol < input_npol; ipol++)
          {
            for (unsigned idim=0; idim < input_ndim; idim++)
            {
              if (do_decay)
              {
                tmp = ((*in_data) + offsets[ipol][ichan]) * scales[ipol][ichan];
                decay_offset[ipol][ichan] = (tmp + decay_offset[ipol][ichan]*decay_constant) / (1.0 + decay_constant);
                (*out_data) = tmp - decay_offset[ipol][ichan];
              }
              else {
                (*out_data) = ((*in_data) + offsets[ipol][ichan]) * scales[ipol][ichan];
              }
              in_data++;
              out_data++;
            }
          }
        }
      }
      break;
    }

    case TimeSeries::OrderFPT:
    {
      for (unsigned ipol=0; ipol < input_npol; ipol++)
      {
        for (unsigned ichan=0; ichan < input_nchan; ichan++)
        {
          const float* in_data = input->get_datptr (ichan, ipol);
          float* out_data = output->get_datptr (ichan, ipol);

          const float the_offset = offsets[ipol][ichan];
          const float the_scale = scales[ipol][ichan];
          uint64_t ival = start_dat * input_ndim;
          for (uint64_t idat=start_dat; idat < end_dat; idat++)
          {
            for (unsigned idim=0; idim < input_ndim; idim++)
            {
              out_data[ival] = (in_data[ival] + the_offset) * the_scale;
              ival++;
            }
          }
        }
      }
      break;
    }
    default:
      throw Error (InvalidState, "dsp::Rescale::rescale ",
          "Requires data in TFP or FPT order");
  }
}

//! Get the epoch of the last scale/offset update
MJD dsp::Rescale::get_update_epoch () const
{
  if (engine)
    return engine->get_update_epoch();

  return update_epoch;
}

const float* dsp::Rescale::get_offset (unsigned ipol) const
{
  if (engine)
    return engine->get_offset(ipol);

  return calculator->get_offset(ipol);
}

const float* dsp::Rescale::get_scale (unsigned ipol) const
{
  if (engine)
    return engine->get_scale(ipol);

  return calculator->get_scale(ipol);
}

const double* dsp::Rescale::get_mean (unsigned ipol) const
{
  if (engine)
    return engine->get_freq_total(ipol);

  return calculator->get_variance(ipol);
}

const double* dsp::Rescale::get_variance (unsigned ipol) const
{
  if (engine)
    return engine->get_freq_squared_total(ipol);

  return calculator->get_variance(ipol);
}

uint64_t dsp::Rescale::get_nsample () const
{
  return nsample;
}

const float* dsp::Rescale::get_time (unsigned ipol) const
{
  return calculator->get_time(ipol);
}

void dsp::Rescale::set_engine(Engine *_engine)
{
  engine = _engine;
  if (calculator)
    engine->set_calculator(calculator);
}

void dsp::Rescale::fire_scales_updated(const dsp::TimeSeries* input, uint64_t start_dat)
{
  ASCIIObservation observation(input);

  auto sample_offset = static_cast<uint64_t>(input->get_input_sample()) + start_dat;

  update_record record
  {
    sample_offset,
    // the number of values used to determine the scales and offsets include all dimensions per sample
    isample * static_cast<uint64_t>(input->get_ndim()),
    calculator->get_scales(),
    calculator->get_offsets(),
    observation
  };

  scales_updated(record);
}

const float* dsp::Rescale::ScaleOffsetCalculator::get_scale (unsigned ipol) const
{
  assert (ipol < scale.size());
  return &(scale[ipol][0]);
}

const float* dsp::Rescale::ScaleOffsetCalculator::get_offset (unsigned ipol) const
{
  assert (ipol < offset.size());
  return &(offset[ipol][0]);
}

const float* dsp::Rescale::ScaleOffsetCalculator::get_time (unsigned ipol) const
{
  assert (ipol < time_total.size());
  return &(time_total[ipol][0]);
}

void dsp::Rescale::set_calculator(ScaleOffsetCalculator* _calculator)
{
  if (engine)
    engine->set_calculator(_calculator);

  calculator = _calculator;
}

const dsp::Rescale::ScaleOffsetCalculator* dsp::Rescale::get_calculator() const
{
  if (engine) {
    return engine->get_calculator();
  }
  return calculator;
}
