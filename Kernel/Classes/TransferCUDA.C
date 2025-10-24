/***************************************************************************
 *
 *   Copyright (C) 2010 - 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferCUDA.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"

#include <iostream>
#include <string.h>
#include <assert.h>

using namespace std;

void dsp::TransferCUDA::copy (void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
{
  assert (dst != nullptr);
  assert (src != nullptr);
  assert (count > 0);

  if (kind == cudaMemcpyHostToHost)
  {
    memcpy(dst, src, count);
    return;
  }

  cudaError error;
  if (stream)
    error = cudaMemcpyAsync (dst, src, count, kind, stream);
  else
    error = cudaMemcpy (dst, src, count, kind);

  if (error != cudaSuccess)
  {
    if (dsp::Operation::verbose)
      std::cerr << "dsp::TransferCUDA::copy failed with error " << cudaGetErrorString(error) << endl;

    throw Error (InvalidState, "dsp::TransferCUDA::copy", cudaGetErrorString (error));
  }

  /*
    2025-03-17 WvS This synchronize is necessary after either device-to-host or host-to-device transfer
    1) after device-to-host, the CPU should wait until the memory has been transferred before reading from it
    2) after host-to-device, the CPU should wait unthe the memory has been transferred before writing to it
  */

  if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyHostToDevice)
  {
    if (verbose)
    {
      std::cerr << "dsp::TransferCUDA::copy synchronizing after "
           << ((kind == cudaMemcpyDeviceToHost) ? "device-to-host" : "host-to-device") << " transfer" << endl;
    }
    if (stream)
      cudaStreamSynchronize(stream);
    else
      cudaDeviceSynchronize();
  }
}

dsp::TransferCUDA::TransferCUDA(cudaStream_t _stream)
  : Transformation<TimeSeries,TimeSeries> ("CUDA::Transfer", outofplace)
{
  stream = _stream;
  kind = cudaMemcpyHostToDevice;
}

void dsp::TransferCUDA::transformation ()
{
  if (verbose)
    cerr << "dsp::TransferCUDA::transformation" << endl;

  prepare ();

  // only transfer data if there is valid data to transfer
  if (input->get_ndat() == 0)
  {
    if (verbose)
      cerr << "dsp::TransferCUDA::transformation skipping transfer as ndat=" << input->get_ndat() << endl;
    return;
  }

  /*
    2025-03-17 WvS This synchronize does not seem necessary.
    The copy is asynchronous only on the host/CPU, and will be scheduled to take place only after all currently
    scheduled GPU kernels have completed.  These following block of code is kept only because it was here and
    it doesn't really hurt to keep it (for now).
  */
  if (kind == cudaMemcpyHostToDevice)
  {
    if (verbose)
      cerr << "dsp::TransferCUDA::transformation synchronizing before host-to-device transfer" << endl;
    if (stream)
      cudaStreamSynchronize(stream);
    else
      cudaDeviceSynchronize();
  }

  if (verbose)
  {
    cerr << "dsp::TransferCUDA::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim()
         << " nchan=" << input->get_nchan() << " npol=" << input->get_npol();
    if (input->get_order() == TimeSeries::OrderFPT)
    {
      if (input->get_npol() > 1)
        cerr << " span=" << input->get_datptr (0,1) - input->get_datptr(0,0);
      cerr << " offset=" << input->get_datptr(0,0) - (float*)input->internal_get_buffer() << endl;
    }
    else
      cerr << endl;
  }

  if (verbose)
    cerr << "dsp::TransferCUDA::transformation xfer data" << endl;

  copy (output->internal_get_buffer(), input->internal_get_buffer(), input->internal_get_size(), kind, stream);

  if (verbose)
  {
    cerr << "dsp::TransferCUDA::transformation output ndat="
         << output->get_ndat() << " ndim=" << output->get_ndim()
         << " nchan=" << output->get_nchan() << " npol=" << output->get_npol();
    if (output->get_order() == TimeSeries::OrderFPT)
    {
      if (output->get_npol() > 1)
        cerr << " span=" << output->get_datptr (0, 1) - output->get_datptr(0,0);
      cerr << " offset=" << output->get_datptr(0,0) - (float*)output->internal_get_buffer() << endl;
    }
    else
      cerr << endl;
  }
}

void dsp::TransferCUDA::prepare ()
{
  output->set_match( const_cast<TimeSeries*>(input.get()) );
  output->internal_match( input );

  // WeightedTimeSeries will copy_all_weights during copy_configuration
  output->copy_configuration( input );

  if (verbose)
    cerr << "dsp::TransferCUDA::prepare input->ndat=" << input->get_ndat() << " output->ndat=" << output->get_ndat() << endl;
}
