/***************************************************************************
 *
 *   Copyright (C) 2024 by William Gauvin, Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/GenericVoltageDigitizer.h"
#include "dsp/TFPOffset.h"
#include "dsp/Scratch.h"
#include "dsp/WeightedTimeSeries.h"
#include "true_math.h"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/GenericVoltageDigitizerCUDA.h"
#endif

#include "debug.h"

#include <cmath>
#include <cstring>
#include <bitset>

dsp::GenericVoltageDigitizer::GenericVoltageDigitizer() : Digitizer("GenericVoltageDigitizer")
{
}

void dsp::GenericVoltageDigitizer::set_engine(Engine* _engine)
{
  if (verbose)
  {
    std::cerr << "dsp::GenericVoltageDigitizer::set_engine" << std::endl;
  }

  engine = _engine;
}

void dsp::GenericVoltageDigitizer::set_device(Memory* memory)
{
  if (verbose)
  {
    std::cerr << "dsp::GenericVoltageDigitizer::set_device()" << std::endl;
  }
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast<CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    cudaStream_t stream = gpu_mem->get_stream();
    if (verbose)
    {
      std::cerr << "dsp::GenericVoltageDigitizer::set_engine()" << std::endl;
    }
    set_engine (new CUDA::GenericVoltageDigitizerEngine(stream));

    // create a new Scratch that uses device memory
    Scratch* gpu_scratch = new Scratch;
    gpu_scratch->set_memory (gpu_mem);
    set_scratch (gpu_scratch);
  }
#endif

  if (engine)
    engine->setup (this);
  else
    Digitizer::set_device (memory);

  device_prepared = true;
}

void dsp::GenericVoltageDigitizer::set_output_weights(dsp::BitSeries* _output_weights)
{
  if (verbose)
    std::cerr << "dsp::GenericVoltageDigitizer set_output_weights _bitseries_weights: "<< _output_weights << std::endl;
  output_weights = _output_weights;
}

void dsp::GenericVoltageDigitizer::prepare ()
{
  if (verbose)
    std::cerr << "dsp::GenericVoltageDigitizer::prepare Digitizer::prepare()" << std::endl;

  Digitizer::prepare ();

  weighted_input = dynamic_cast<const WeightedTimeSeries*>(input.get());

  if (!weighted_input)
  {
    if (verbose)
      std::cerr << "dsp::GenericVoltageDigitizer::prepare input is not a WeightedTimeSeries" << std::endl;

    return;
  }

  if (!output_weights)
  {
    if (verbose)
      std::cerr << "dsp::GenericVoltageDigitizer::prepare no output_weights BitSeries to copy to" << std::endl;

    weighted_input = nullptr;
    return;
  }

  if (weighted_input->get_ndat_per_weight() == 0)
  {
    if (verbose)
      std::cerr << "dsp::GenericVoltageDigitizer::prepare WeightedTimeSeries::get_ndat_per_weight not configured" << std::endl;

    weighted_input = nullptr;
    return;
  }

  if (verbose)
    std::cerr << "dsp::GenericVoltageDigitizer::prepare prepare_weights()" << std::endl;

  prepare_weights();
}

void dsp::GenericVoltageDigitizer::prepare_weights ()
{
  if (!weighted_input)
    return;

  const auto nchan_weight = weighted_input->get_nchan_weight();
  const auto npol_weight = weighted_input->get_npol_weight();
  const auto ndat_per_weight = weighted_input->get_ndat_per_weight();

  if (verbose)
  {
    std::cerr << "dsp::GenericVoltageDigitizer::prepare_weights nchan_weight: " << nchan_weight << std::endl;
    std::cerr << "dsp::GenericVoltageDigitizer::prepare_weights npol_weight: " << npol_weight << std::endl;
    std::cerr << "dsp::GenericVoltageDigitizer::prepare_weights ndat_per_weight: " << ndat_per_weight << std::endl;
  }

  // Ensure the correct configuration and size
  output_weights->copy_configuration(weighted_input);
  output_weights->set_nchan(nchan_weight);
  output_weights->set_npol(npol_weight);

  // Weights are 16-bit and real-valued
  output_weights->set_nbit(16);
  output_weights->set_ndim(1);

  // Weights have a reduced sampling rate
  output_weights->set_rate(input->get_rate() / ndat_per_weight);
}

void dsp::GenericVoltageDigitizer::copy_weights ()
{
  if (!weighted_input)
    return;

  const auto nchan_weight = weighted_input->get_nchan_weight();
  const auto npol_weight = weighted_input->get_npol_weight();
  const auto nweights = weighted_input->get_nweights();
  const auto ndat_per_weight = weighted_input->get_ndat_per_weight();

  output_weights->resize (nweights);

  if (verbose)
  {
    std::cerr << "dsp::GenericVoltageDigitizer::copy_weights ndat_per_weight: " << ndat_per_weight << std::endl;
    std::cerr << "dsp::GenericVoltageDigitizer::copy_weights nchan_weight: " << nchan_weight << std::endl;
    std::cerr << "dsp::GenericVoltageDigitizer::copy_weights npol_weight: " << npol_weight << std::endl;
    std::cerr << "dsp::GenericVoltageDigitizer::copy_weights nweights: " << nweights << std::endl;
  }

  auto out_weights_ptr = reinterpret_cast<uint16_t*>(output_weights->get_rawptr());

  if (verbose)
    std::cerr << "out_weights_ptr: " << out_weights_ptr << std::endl;

  if (verbose)
    std::cerr << "dsp::GenericVoltageDigitizer::copy_weights transposing from FPT to TFP" << std::endl;

  for (unsigned ichan = 0; ichan < nchan_weight; ichan++)
  {
    for (unsigned ipol = 0; ipol < npol_weight; ipol++)
    {
      const uint16_t* in_weights_ptr = weighted_input->get_weights(ichan, ipol);
      for (uint64_t iweight = 0; iweight < nweights; iweight++)
      {
        // output weights in TFP (iweight, ichan, ipol) order
        auto output_idx = iweight * (nchan_weight * npol_weight) + ichan * (npol_weight) + ipol;
        out_weights_ptr[output_idx] = in_weights_ptr[iweight];

        DEBUG("iweight=" << iweight << " ichan="<< ichan << " ipol="<< ipol << " weight["<<output_idx<<"]: " << out_weights_ptr[output_idx]);
      }
    }
  }

  if (verbose)
    std::cerr << "dsp::GenericVoltageDigitizer::copy_weights transpose and copy complete" << std::endl;
}

bool dsp::GenericVoltageDigitizer::get_device_supported(Memory* memory) const
{
  if (verbose)
    std::cerr << "dsp::GenericVoltageDigitizer::get_device_supported memory=" << (void *) memory << std::endl;
#ifdef HAVE_CUDA
  if (dynamic_cast<CUDA::DeviceMemory*>(memory) != nullptr)
    return true;
#endif
  return Digitizer::get_device_supported(memory);
}

void dsp::GenericVoltageDigitizer::set_nbit(int bit_width)
{
  digi_mean = dsp::GenericVoltageDigitizer::get_default_digi_mean(bit_width);
  digi_scale = dsp::GenericVoltageDigitizer::get_default_digi_scale(bit_width);

  nbit = bit_width;
  int digi_limit = std::pow(2, nbit - 1);
  digi_min = -1 * digi_limit;
  digi_max = digi_limit - 1;
  if (verbose)
  {
    std::cerr << "dsp::GenericVoltageDigitizer::set_nbit nbit=" << nbit << " digi_mean=" << digi_mean
      << " digi_scale=" << digi_scale << " digi_min=" << digi_min << " digi_max=" << digi_max << std::endl;
  }
}

float dsp::GenericVoltageDigitizer::get_default_digi_mean(int bit_width)
{
  switch (bit_width)
  {
  case 2:
  case 4:
  case 8:
  case 16:
    return -0.5;
  case 1:
  case -32:
    return 0;
  default:
    throw Error(InvalidState, "dsp::GenericVoltageDigitizer::get_default_digi_mean",
                "bit_width=%i not supported", bit_width);
    break;
  }
}

/*!
  The default values are the optimal values derived in column E of
  https://docs.google.com/spreadsheets/d/1F01T1KAoSTZOaW33wYVq6EZJk_xuwu3xuV2oohFhBrA/edit?gid=0#gid=0
*/
float dsp::GenericVoltageDigitizer::get_default_digi_scale(int bit_width)
{
  switch (bit_width)
  {
  case 1:
    return 1.0;
  case 2:
    return 1.03;
  case 4:
    return 3.14;
  case 8:
    return 10.1;    // 6 bits of signal + 2 bits of overhead for RFI
  case 16:
    return 1106.4;  // 14 bits of signal + 2 bits of overhead for RFI
  case -32:
    return 1.0;
  default:
    throw Error(InvalidState, "dsp::GenericVoltageDigitizer::get_default_digi_scale",
                "bit_width=%i not supported", bit_width);
    break;
  }
}

void dsp::GenericVoltageDigitizer::pack()
{
  if (digi_scale == 0)
  {
    throw Error(InvalidState, "dsp::GenericVoltageDigitizer::pack", "set_nbit has not been called");
  }

  // the number of time samples
  const uint64_t ndat = input->get_ndat();

  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // number of polns
  const unsigned npol = input->get_npol();

  // number of dims
  const unsigned ndim = input->get_ndim();

  if (ndim != 2)
    throw Error(InvalidState, "dsp::GenericVoltageDigitizer::pack",
                "cannot handle ndim=%d", ndim);

  if (ndat == 0)
    return;

  // ensure the number of samples wholly fits within an integer byte
  static constexpr int bits_per_byte = 8;
  if (nbit < bits_per_byte && nbit >= 1)
  {
    int samples_per_byte = bits_per_byte / nbit;
    int nvalues_to_pack = ndat * nchan * npol * ndim;
    if (nvalues_to_pack % samples_per_byte != 0)
    {
      throw Error(InvalidParam, "dsp::GenericVoltageDigitizer::pack",
        "total number of values to pack [%d] was not a multiple of samples_per_byte [%d]",
        nvalues_to_pack, samples_per_byte);
    }
  }

  // Also apply any existing scale factors (note, Rescale will set the
  // input scale to 1.0 if it has been applied to the data).
  const float effective_scale = digi_scale / (input->get_scale() * scale_fac);

  int samp_per_byte = bits_per_byte / nbit;
  if (!samp_per_byte)
    samp_per_byte = 1;

  // bit masking is only used for nbit = 1, 2, or 4
  const unsigned char mask = (nbit < 8) ? static_cast<unsigned char>(pow(2, nbit) - 1) : 0xff;
  size_t scratch_needed = 0;
  float* scratch_space = nullptr;

  if (input->get_order() == dsp::TimeSeries::OrderFPT)
  {
    scratch_needed = input->internal_get_size();
    scratch_space = scratch->space<float>(scratch_needed);
    if (verbose)
    {
      std::cerr << "dsp::GenericVoltageDigitizer::pack scratch_needed=" << scratch_needed << " scratch_space=" << reinterpret_cast<void*>(scratch_space) << std::endl;
    }
  }

  if (engine)
  {
    if (input->get_order() == dsp::TimeSeries::OrderFPT)
    {
      engine->set_scratch_space(scratch_space, scratch_needed);
    }

    if (verbose)
      std::cerr << "dsp::GenericVoltageDigitizer::pack using Engine" << std::endl;
    engine->pack(
      input,
      output,
      nbit,
      digi_min,
      digi_max,
      digi_mean,
      digi_scale,
      effective_scale,
      samp_per_byte
    );

    if (weighted_input && output_weights)
      engine->copy_weights(weighted_input, output_weights);

    return;
  }
  copy_weights();

  if (nbit == -32)
  {
    pack_float ();
    return;
  }

  const float * tfp_input = nullptr;

  // To simplify packing of FPT to TFP, transpose first if required
  if (input->get_order() == TimeSeries::OrderFPT)
  {
    for (unsigned ichan = 0; ichan < nchan; ichan++)
    {
      for (unsigned ipol = 0; ipol < npol; ipol++)
      {
        const float *inptr = input->get_datptr(ichan, ipol);
        uint64_t idx = 0;
        for (uint64_t idat = 0; idat < ndat; idat++)
        {
          for (unsigned idim = 0; idim < ndim; idim++)
          {
            uint64_t odx = ((idat * nchan + ichan) * npol + ipol) * ndim + idim;
            scratch_space[odx] = inptr[idx++];
          }
        }
      }
    }
    tfp_input = reinterpret_cast<const float *>(scratch_space);
  }
  else
  {
    tfp_input = input->get_dattfp();
  }

  int byte_sample_idx = 0;
  uint64_t idx = 0;

  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    char *outptr;

    if (nbit == 16)
    {
      outptr = reinterpret_cast<char *>(output->get_rawptr()) + 2 * (idat * nchan * npol * ndim);
    }
    else
    {
      outptr = reinterpret_cast<char *>(output->get_rawptr()) + (idat * nchan * npol * ndim) / samp_per_byte;
    }

    for (unsigned ichan = 0; ichan < nchan; ichan++)
    {
      for (unsigned ipol = 0; ipol < npol; ipol++)
      {
        for (unsigned idim = 0; idim < ndim; idim++)
        {
          int result;
          if (nbit == 1)
          {
            // if value is 0.0 then value is 1 if -0.0 it will be 0
            result = !true_signbit_float(tfp_input[idx]);
          }
          else
          {
            const float scaled = std::round(fmaf(tfp_input[idx], effective_scale, digi_mean));
            result = static_cast<int>(scaled);

            result = std::max(result, digi_min);
            result = std::min(result, digi_max);
          }

          // WvS - 2024-07-11 consider using templates to move the switch statement out of the inner loop
          switch (nbit)
          {
          case 1:
          case 2:
          case 4:
            byte_sample_idx = idx % samp_per_byte;

            if (byte_sample_idx == 0)
              (*outptr) = (unsigned char)0;

            *outptr |= (static_cast<char>(result) & mask) << (byte_sample_idx * nbit);

            // check if the next sample through would be in the next byte
            if (byte_sample_idx == (samp_per_byte - 1))
              outptr++;

            break;
          case 8:
            *outptr = static_cast<char>(result);
            outptr++;
            break;
          case 16:
            *(reinterpret_cast<int16_t *>(outptr)) = static_cast<int16_t>(result);
            outptr++;
            outptr++;
            break;
          }
          idx++;
        } // dim
      }   // poln
    }     // chan
  }       // time
}

void dsp::GenericVoltageDigitizer::pack_float () try
{
  // the number of frequency channels
  const unsigned nchan = input->get_nchan();

  // the number of time samples
  const uint64_t ndat = input->get_ndat();

  // number of polarizations
  const unsigned npol = input->get_npol();

  // number of dimensions
  const unsigned ndim = input->get_ndim();

  // scale factor
  const float effective_scale = digi_scale / (input->get_scale() * scale_fac);
  float* outptr = reinterpret_cast<float*>( output->get_rawptr() );

  switch (input->get_order())
  {
  case TimeSeries::OrderTFP:
  {
    // output order = input order ... almost a memcpy, but for the effective scale
    const float* inptr = input->get_dattfp();
    const uint64_t nfloat = ndat * nchan * npol * ndim;
    for (uint64_t ifloat=0; ifloat < nfloat; ifloat++)
    {
      outptr[ifloat] = inptr[ifloat] * effective_scale;
    }
    return;
  }
  case TimeSeries::OrderFPT:
  {
    // input and output have the same dimensions
    TFPOffset output_offset (input);

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        const float* inptr = input->get_datptr( ichan, ipol );

        for (uint64_t idat=0; idat < ndat; idat++)
        {
          for (unsigned idim=0; idim < ndim; idim++)
          {
            outptr[output_offset(idat,ichan,ipol) + idim] = inptr[idat*ndim + idim] * effective_scale;
          }
        }
      }
    }
    return;
  }

  default:
    throw Error (InvalidState, "dsp::GenericVoltageDigitizer::pack_float",
		 "Can only operate on data ordered FTP or FPT.");
  }
}
catch (Error& error)
{
  throw error += "dsp::GenericVoltageDigitizer::pack_float";
}

void dsp::GenericVoltageDigitizer::Engine::setup (GenericVoltageDigitizer* user)
{
}
