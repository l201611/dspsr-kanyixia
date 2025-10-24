//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SKAParallelUnpacker.h"
#include "dsp/ParallelInput.h"
#include "dsp/ASCIIObservation.h"
#include "dsp/WeightedTimeSeries.h"

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/SKAParallelUnpackerCUDA.h"
#endif

#include "dsp/BitTable.h"

#include "true_math.h"
#include <assert.h>

using namespace std;

dsp::SKAParallelUnpacker::SKAParallelUnpacker () : ParallelUnpacker ("SKAParallelUnpacker") { }

dsp::SKAParallelUnpacker::~SKAParallelUnpacker () { }

void dsp::SKAParallelUnpacker::set_engine (Engine* _engine)
{
  if (verbose)
  {
    cerr << "dsp::SKAParallelUnpacker::set_engine" << endl;
  }

  engine = _engine;
}

void dsp::SKAParallelUnpacker::prepare ()
{
  if (verbose)
    cerr << "dsp::SKAParallelUnpacker::prepare()"<< endl;
  reserve();
}

bool dsp::SKAParallelUnpacker::get_device_supported (Memory* memory) const
{
  if (verbose)
    cerr << "dsp::SKAParallelUnpacker::get_device_supported memory=" << (void *) memory << endl;
#ifdef HAVE_CUDA
  if (dynamic_cast<CUDA::DeviceMemory*>(memory) != nullptr)
    return true;
#endif
  return ParallelUnpacker::get_device_supported(memory);
}

void dsp::SKAParallelUnpacker::set_device (Memory* memory)
{
  if (verbose)
    cerr << "dsp::SKAParallelUnpacker::set_device()" << endl;
#if HAVE_CUDA
  CUDA::DeviceMemory * gpu_mem = dynamic_cast< CUDA::DeviceMemory*>( memory );
  if (gpu_mem)
  {
    cudaStream_t stream = gpu_mem->get_stream();
    cout << "dsp::SKAParallelUnpacker::set_engine()" << endl;
    set_engine (new CUDA::SKAParallelUnpackerEngine(stream));
  }
#endif

  if (engine)
    engine->setup (this);
  else
    ParallelUnpacker::set_device (memory);

  device_prepared = true;
}

void dsp::SKAParallelUnpacker::configure (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::SKAParallelUnpacker::configure" << endl;

  if (configured)
    return;

  // set standard observation configuration parameters
  ndim = observation->get_ndim();
  npol = observation->get_npol();
  nbit = observation->get_nbit();
  nchan = observation->get_nchan();
  machine = string(observation->get_machine());
  if (verbose)
    cerr << "dsp::SKAParallelUnpacker::configure ndim=" << ndim << " npol=" << npol
              << " nbit=" << nbit << " nchan=" << nchan << " machine=" << machine << endl;

  // set custom configuration parameters that are bespoke to the SKAParallelUnpacker
  const ASCIIObservation * info = dynamic_cast<const ASCIIObservation *>(observation);
  if (info)
  {
    info->custom_header_get ("UDP_NSAMP", "%u", &nsamp_per_packet);
    info->custom_header_get ("UDP_NCHAN", "%u", &nchan_per_packet);
    info->custom_header_get ("WT_NSAMP", "%u", &nsamp_per_weight);
    try {
      uint32_t wt_valid{0};
      info->custom_header_get ("WT_VALID", "%u", &wt_valid);
      weights_valid = (wt_valid == 1);
    } catch (Error& error) {
      weights_valid = false;
    }

    if (verbose)
      cerr << "dsp::SKAParallelUnpacker::configure nsamp_per_packet=" << nsamp_per_packet << " nchan_per_packet=" << nchan_per_packet << " nsamp_per_weight=" << nsamp_per_weight << endl;
    npackets_per_heap = nchan / nchan_per_packet;
    if (nchan % nchan_per_packet != 0)
    {
      throw Error (InvalidState, "dsp::SKAParallelUnpacker::configure",
	                "nchan [%u] was not a multiple of UDP_NCHAN [%u]", nchan, nchan_per_packet);
    }

    weights_packet_stride = scale_nbyte + ((nchan_per_packet * weight_nbyte * nsamp_per_packet) / nsamp_per_weight);
    if (verbose)
      cerr << "dsp::SKAParallelUnpacker::configure npackets_per_heap=" << npackets_per_heap
                << " weights_packet_stride=" << weights_packet_stride << endl;
    configured = true;
  }
  else
  {
    throw Error(InvalidState, "dsp::SKAParallelUnpacker::configure", "Could not interpret Observation as ASCIIObservation");
  }
}

bool dsp::SKAParallelUnpacker::matches (const Observation* observation) const try
{
  bool valid = (
    (observation->get_ndim() == 2) &&
    (observation->get_npol() == 2) &&
    (
      ((observation->get_machine() == "LowCBF") && (observation->get_nbit() == 16)) ||
      ((observation->get_machine() == "MidCBF") && ((observation->get_nbit() == 16) || (observation->get_nbit() == 8)))
    )
  );

  if (verbose)
    cerr << "dsp::SKAParallelUnpacker::matches ndim=" << observation->get_ndim()
      << " npol=" << observation->get_npol()
      << " nbit=" << observation->get_nbit()
      << " machine=" << observation->get_machine() << " valid=" << valid << endl;
  return valid;
}
catch (Error& error)
{
  return false;
}

void dsp::SKAParallelUnpacker::match (const Observation* observation)
{
  if (verbose)
    cerr << "dsp::SKAParallelUnpacker::match" << endl;
  configure(observation);
}

void dsp::SKAParallelUnpacker::reserve ()
{
  ParallelUnpacker::reserve ();

  auto weighted = dynamic_cast<WeightedTimeSeries*>(output.get());
  if (weighted)
  {
    if (verbose)
      cerr << "dsp::SKAParallelUnpacker::reserve configuring WeightedTimeSeries" << endl;
    weighted->set_ndat_per_weight (nsamp_per_weight);
    weighted->set_npol_weight (1);
    weighted->set_nchan_weight (output->get_nchan());
    weighted->resize(output->get_ndat());
  }
}

void dsp::SKAParallelUnpacker::unpack ()
{
  if (verbose)
    cerr << "dsp::SKAParallelUnpacker::unpack input=" << input.get() << endl;

  if (input->size() != 2)
    throw Error (InvalidState, "dsp::SKAParallelUnpacker::unpack",
	               "Number of bit series was %d, expecting 2", input->size());

  if (unpackers.size() != 0)
    throw Error (InvalidState, "dsp::SKAParallelUnpacker::unpack",
	               "Number of unpackers was %d, expecting 0", unpackers.size());

  if (!configured)
  {
    throw Error (InvalidState, "dsp::SKAParallelUnpacker::unpack",
                "dont yet know how to get access of the observation from the data bitseries.");
  }

  const BitSeries* data = input->at(0);
  const BitSeries* weights = input->at(1);

  if (engine)
  {
    if (verbose)
      std::cerr << "dsp::SKAParallelUnpacker::unpack using Engine" << std::endl;
    engine->unpack(data, weights, output, nsamp_per_packet, nchan_per_packet, nsamp_per_weight, weights_valid);
    return;
  }

  const uint64_t ndat = data->get_ndat();
  const uint32_t nheaps = ndat / nsamp_per_packet;
  const uint32_t npackets_per_heap = nchan / nchan_per_packet;

  const unsigned char* weights_from = weights->get_rawptr();

  // unpack weights into WeightedTimeSeries on CPU
  auto weighted = dynamic_cast<WeightedTimeSeries*>(output.get());
  if (weighted)
  {
    if (verbose)
      std::cerr << "dsp::SKAParallelUnpacker::unpack unpacking weights into WeightedTimeSeries" << std::endl;

    // + sizeof(float) to skip the scale factor at the start of each heap
    auto weights_base = weights_from + sizeof(float);

    for (uint32_t iheap=0; iheap<nheaps; iheap++)
    {
      unsigned outchan = 0;

      for (uint32_t ipacket=0; ipacket<npackets_per_heap; ipacket++)
      {
        auto weights_ptr = reinterpret_cast<const uint16_t*>(weights_base);
        weights_base += weights_packet_stride;

        for (uint32_t ichan=0; ichan<nchan_per_packet; ichan++)
        {
          auto weights_out = weighted->get_weights(outchan);
          if (weights_valid)
            weights_out[iheap] = weights_ptr[ichan];
          else
            weights_out[iheap] = 1;
          outchan ++;
        }
      }
    }
  }

  const unsigned char * data_from = data->get_rawptr();

  if (weights->get_size() < nheaps * npackets_per_heap * weights_packet_stride)
    throw Error (InvalidState, "dsp::SKAParallelUnpacker::unpack",
                "weights->size=%lu is less than %lu", weights->get_size(),
                nheaps * npackets_per_heap * weights_packet_stride);

  if (nbit == 8)
  {
    unpack_samples(reinterpret_cast<const int8_t*>(data_from), weights_from, nheaps);
  }
  else
  {
    unpack_samples(reinterpret_cast<const int16_t*>(data_from), weights_from, nheaps);
  }
}

float dsp::SKAParallelUnpacker::get_scale_factor (const unsigned char * weights, uint32_t packet_number)
{
  auto * weights_ptr = reinterpret_cast<const float *>(weights + (packet_number * weights_packet_stride));
  return *weights_ptr;
}

void dsp::SKAParallelUnpacker::Engine::setup (SKAParallelUnpacker* user)
{
}
