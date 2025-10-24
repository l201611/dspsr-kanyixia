//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2018-2025 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/ASCIIObservation.h"
#include "dsp/UWBFloatUnpacker.h"

#include "Error.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/UWBFloatUnpackerCUDA.h"
#include <cuda_runtime.h>
#endif

#include <errno.h>
#include <string.h>

using namespace std;

static void* const undefined_stream = (void *) -1;

dsp::UWBFloatUnpacker::UWBFloatUnpacker (const char* _name) : Unpacker (_name)
{
  if (verbose)
    cerr << "dsp::UWBFloatUnpacker ctor" << endl;

  input_order = "unknown";
  first_block = true;
  npol = 2;
  ndim = 2;
}

dsp::UWBFloatUnpacker::~UWBFloatUnpacker ()
{
}

dsp::UWBFloatUnpacker * dsp::UWBFloatUnpacker::clone () const
{
  return new UWBFloatUnpacker (*this);
}

//! Return true if the unpacker support the specified output order
bool dsp::UWBFloatUnpacker::get_order_supported (TimeSeries::Order order) const
{
  return (order == TimeSeries::OrderFPT);
}

//! Set the order of the dimensions in the output TimeSeries
void dsp::UWBFloatUnpacker::set_output_order (TimeSeries::Order order)
{
  output_order = order;
}

void dsp::UWBFloatUnpacker::set_engine (Engine* _engine)
{
  if (verbose)
    cerr << "dsp::UWBFloatUnpacker::set_engine()" << endl;
  engine = _engine;
}

//! Return true if the unpacker can operate on the specified device
bool dsp::UWBFloatUnpacker::get_device_supported (Memory* memory) const
{
  if (verbose)
    cerr << "dsp::UWBFloatUnpacker::get_device_supported()" << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    CUDA::UWBFloatUnpackerEngine * tmp = new CUDA::UWBFloatUnpackerEngine(0);
    return tmp->get_device_supported (memory);
  }
#endif
  return Unpacker::get_device_supported(memory);
}

//! Set the device on which the unpacker will operate
void dsp::UWBFloatUnpacker::set_device (Memory* memory)
{
  if (verbose)
    cerr << "dsp::UWBFloatUnpacker::set_device" << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    cudaStream_t stream = gpu_mem->get_stream();
    if (verbose)
      cerr << "dsp::UWBFloatUnpacker::set_device Device Memory supported with stream " << (void *) stream << endl;
    set_engine (new CUDA::UWBFloatUnpackerEngine(stream));
  }
#endif
  if (engine)
  {
    if (verbose)
      cerr << "dsp::UWBFloatUnpacker::set_device engine->set_device()" << endl;
    engine->set_device(memory);
    if (verbose)
      cerr << "dsp::UWBFloatUnpacker::set_device engine->setup()" << endl;
    engine->setup();
  }
  else
    Unpacker::set_device (memory);
  if (verbose)
    cerr << "dsp::UWBFloatUnpacker::set_device device prepared" << endl;
  device_prepared = true;
}

bool dsp::UWBFloatUnpacker::matches (const Observation* observation)
{
  return (observation->get_machine() == "UWB" || observation->get_machine()== "Medusa" || observation->get_machine()== "Apollo")
    && observation->get_nchan() == 1
    && (observation->get_npol() == 2 || observation->get_npol() == 1)
    && (observation->get_nbit() == 32);
}

void dsp::UWBFloatUnpacker::unpack ()
{
  unsigned ndat = input->get_ndat ();
  unsigned npol = input->get_npol ();
  unsigned nchan = input->get_nchan ();
  unsigned ndim = input->get_ndim ();

  if (verbose)
    cerr << "dsp::UWBFloatUnpacker::unpack ndat=" << ndat << " npol=" << npol
         << " nchan=" << nchan << " ndim=" << ndim << endl;
  if (ndat == 0)
    return;

  if (input_order == "unknown")
  {
    const Input * in = input->get_loader();
    const Observation * obs = in->get_info();
    const ASCIIObservation * info = dynamic_cast<const ASCIIObservation *>(obs);
    if (!info)
    {
      // default input ordering for processing the full band
      input_order = "SFPT";
    }
    else
    {
      // custom input ordering when processing a zoom band
      char parsed_order[32];
      info->custom_header_get("ORDER", "%s", parsed_order);
      input_order = std::string(parsed_order);
    }
  }

  if (engine)
  {
    if (verbose)
      cerr << "dsp::UWBFloatUnpacker::unpack using Engine" << endl;
    engine->unpack(input, output, input_order);
    return;
  }

  // some programs (digifil) do not call set_device
  if ( ! device_prepared )
    set_device ( Memory::get_manager ());

  float * from = (float *) input->get_rawptr();

  // special input ordering for (effectively TP) ordered data from zoom-bands
  if (input_order == "TFPS")
  {
    float * intos[npol];
    unsigned ichan = 0;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      intos[ipol] = output->get_datptr (ichan, ipol);
    }

    uint64_t idx = 0;
    for (uint64_t idat=0; idat<ndat; idat++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned idim=0; idim<ndim; idim++)
        {
          intos[ipol][idat*ndim+idim] = from[idx];
          idx++;
        }
      }
    }
  }
  // default SFPT ordring (effectively PT) ordered data from sub-bands
  else
  {
    const unsigned ndat_per_block = input->get_loader()->get_resolution();
    uint64_t nblock = ndat / ndat_per_block;

    // number of floats in each pol/chan per block
    size_t block_stride = ndat_per_block * ndim;
    // number of bytes in each pol/chan per block
    size_t to_copy = block_stride * sizeof(float);

    if (verbose)
      cerr << "dsp::UWBFloatUnpacker::unpack nblock=" << nblock
           << " npol=" << npol << " nchan=" << nchan << " block_stride="
           << block_stride << endl;

    for (uint64_t iblock=0; iblock<nblock; iblock++)
    {
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned ichan=0; ichan<nchan; ichan++)
        {
          float * into = output->get_datptr (ichan, ipol) + (iblock * block_stride);
          ::memcpy ((void *) into, (void *) from, to_copy);
          from += block_stride;
        }
      }
    }
  }
}
