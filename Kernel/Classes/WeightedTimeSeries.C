/***************************************************************************
 *
 *   Copyright (C) 2002 - 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "environ.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/MemoryHost.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDA.h"
#endif

#include "Error.h"

#include "debug.h"

#include <stdlib.h>
#include <assert.h>

using namespace std;

dsp::WeightedTimeSeries::WeightedTimeSeries() : TimeSeries()
{
  weights_memory = new MemoryHost;
}

dsp::WeightedTimeSeries::~WeightedTimeSeries()
{
  if (base)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::~WeightedTimeSeries delete base" << endl;
    weights_memory->do_free(base);
    base = weights = nullptr;
  }
}

dsp::WeightedTimeSeries::WeightedTimeSeries (const WeightedTimeSeries& wts) : TimeSeries()
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries copy ctor" << endl;
  operator=(wts);
}

void dsp::WeightedTimeSeries::copy_configuration (const Observation* copy)
{
  TimeSeries::copy_configuration (copy);
  copy_weights (copy);
}

//! Copy the dimensions of another TimeSeries instance
void dsp::WeightedTimeSeries::copy_dimensions (const Observation* copy)
{
  TimeSeries::copy_dimensions (copy);

  auto weighted_copy = dynamic_cast< const WeightedTimeSeries* > (copy);

  if (!weighted_copy)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::copy_dimensions that is not a WeightedTimeSeries" << endl;
    ndat_per_weight = 0;
    return;
  }

  if (weighted_copy == this)
    return;

  if (verbose) cerr << "dsp::WeightedTimeSeries::copy_dimensions that is a WeightedTimeSeries" << endl;

  set_npol_weight  ( weighted_copy->get_npol_weight() );
  set_nchan_weight ( weighted_copy->get_nchan_weight() );
  set_ndat_per_weight ( weighted_copy->get_ndat_per_weight() );
}

void dsp::WeightedTimeSeries::copy_weights (const Observation* copy)
{
  auto weighted_copy = dynamic_cast< const WeightedTimeSeries* > (copy);

  if (!weighted_copy)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::copy_weights "
              "not a WeightedTimeSeries" << endl;
    ndat_per_weight = 0;
    return;
  }

  if (weighted_copy == this)
    return;

  if (verbose) cerr << "dsp::WeightedTimeSeries::copy_weights"
		 " resize weights (ndat=" << get_ndat() << ")" << endl;

  set_npol_weight  ( weighted_copy->get_npol_weight() );
  set_nchan_weight ( weighted_copy->get_nchan_weight() );
  set_ndat_per_weight ( weighted_copy->get_ndat_per_weight() );

  weight_idat = weighted_copy->weight_idat;

  resize_weights ( weighted_copy->get_ndat() );

  copy_all_weights (weighted_copy);
}

void dsp::WeightedTimeSeries::copy_data (const TimeSeries* copy, uint64_t istart, uint64_t ndat)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_data TimeSeries::copy_data" << endl;

  TimeSeries::copy_data (copy, istart, ndat);

  const WeightedTimeSeries* wt = dynamic_cast<const WeightedTimeSeries*>(copy);

  if (!wt)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::copy_data"
              " copy is not a WeightedTimeSeries" << endl;
    return;
  }

  if (wt == this)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::copy_data"
              " copy is equal to this" << endl;
    return;
  }

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_data call copy_weights" << endl;

  copy_weights (wt, istart, ndat);
}

//! Set the number of time samples per weight
void dsp::WeightedTimeSeries::set_ndat_per_weight (unsigned _ndat_per_weight)
{
  ndat_per_weight = _ndat_per_weight;
}

//! Set the number of polarizations with independent weights
void dsp::WeightedTimeSeries::set_npol_weight (unsigned _npol_weight)
{
  npol_weight = _npol_weight;
}

//! Set the number of frequency channels with independent weights
void dsp::WeightedTimeSeries::set_nchan_weight (unsigned _nchan_weight)
{
  nchan_weight = _nchan_weight;
}

//! Get the number of weights
uint64_t dsp::WeightedTimeSeries::get_nweights () const
{
  uint64_t nweights = get_nweights (get_ndat() + weight_idat);

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::get_nweights weight_idat="
         << weight_idat << " nweights=" << nweights << endl;

  if ( weights && (weights + nweights) > (base + weight_subsize) )
  {
    cerr << "dsp::WeightedTimeSeries::get_nweights FATAL ERROR:\n\t"
            "weights=" << weights << " + nweights=" << nweights <<
            " > base=" << base << " + size=" << weight_subsize <<
            " (weight_idat=" << weight_idat <<
            " ndat_per_weight=" << ndat_per_weight << ")" << endl;
    throw Error (InvalidState, "dsp::WeightedTimeSeries::get_nweights",
                 "invalid nweights=%u", nweights);
  }

  return nweights;
}

uint64_t dsp::WeightedTimeSeries::have_nweights () const
{
  return weight_subsize - (weights - base);
}

//! Get the number of weights
uint64_t dsp::WeightedTimeSeries::get_nweights (uint64_t nsamples) const
{
  if (ndat_per_weight == 0)
    return 0;

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::get_nweights ndat_per_weight="
         << ndat_per_weight << " nsamples=" << nsamples << endl;

  uint64_t nweights = nsamples / ndat_per_weight;
  if (nsamples % ndat_per_weight)
    nweights ++;

  return nweights;
}

//! Set the offset of the first time sample in the current weight array
void dsp::WeightedTimeSeries::set_weight_idat (uint64_t _weight_idat)
{
  weight_idat = _weight_idat;
}


dsp::WeightedTimeSeries* dsp::WeightedTimeSeries::clone() const
{
  return new WeightedTimeSeries(*this);
}

dsp::WeightedTimeSeries* dsp::WeightedTimeSeries::null_clone() const
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::null_clone" << endl;

  WeightedTimeSeries* retval = new WeightedTimeSeries;
  retval->null_work (this);
  retval->npol_weight = npol_weight;
  retval->nchan_weight = nchan_weight;
  retval->ndat_per_weight = ndat_per_weight;

  return retval;
}

//! Allocate the space required to store nsamples time samples
/*! This method uses TimeSeries::resize to size the floating point time
  samples array, then resizes the weights array according to the nsamples
  argument as well as the ndat_per_weight, nchan_weight, and npol_weight
  attributes.

  \pre The ndat_per_weight, nchan_weight, and npol_weight attributes must
  be set to their desired values before this method is called.
*/
void dsp::WeightedTimeSeries::resize (uint64_t nsamples)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize (" << nsamples << ")" << endl;

  TimeSeries::resize (nsamples);
  resize_weights (nsamples);
}

void dsp::WeightedTimeSeries::resize_weights (uint64_t nsamples)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize_weights nsamples=" << nsamples << endl;

  uint64_t nweights = get_nweights (weight_idat + nsamples);

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize_weights reserve=" << get_reserve() << endl;

  unsigned reserve_weights = get_nweights(get_reserve() * reserve_kludge_factor);
  nweights += reserve_weights;

  unsigned weight_dimensions = get_npol_weight() * get_nchan_weight();
  uint64_t require = nweights * weight_dimensions;

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize_weights nweights=" << nweights
         << " require=" << require << " have=" << weight_size << endl;

  if (!require || require > weight_size)
  {
    if (base)
    {
      if (verbose)
        cerr << "dsp::WeightedTimeSeries::resize_weights delete" << endl;
      weights_memory->do_free(base); base = weights = 0;
    }

    weight_size = weight_subsize = 0;
  }

  if (!require)
    return;

  if (!weight_size)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::resize_weights new " << require <<endl;

    require += 4 * weight_dimensions;
    nweights += 4;

    base = reinterpret_cast<uint16_t*> (weights_memory->do_allocate(require*sizeof(uint16_t)));
    weight_size = require;
    weight_idat = 0;
    weight_subsize = 0;
  }

  /* BUG FIX: WvS 2025-03-13
     If weight_subsize == 0 (uninitialized) or the current weight_subsize is too large
     (because the weight_dimensions=nchan*npol dimensions of the weights changed)
     then compute a new weight_subsize.
  */

  if (weight_subsize == 0 || weight_dimensions * weight_subsize >= weight_size)
  {
    weight_subsize = weight_size / weight_dimensions;
    assert(weight_subsize >= nweights);
  }

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize_weights reserve"
         << " reserve_kludge_factor=" << reserve_kludge_factor
         << " ndat=" << get_reserve()
         << " -> nwt=" << reserve_weights << endl;

  weights = base + reserve_weights;

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::resize_weights base=" << base
         << " weights=" << weights << endl;
}

//! Match the internal memory layout of another TimeSeries
void dsp::WeightedTimeSeries::internal_match (const TimeSeries* other)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::internal_match this=" << this << " that=" << other << endl;

  TimeSeries::internal_match (other);

  auto like = dynamic_cast<const WeightedTimeSeries*> (other);
  if (like)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::internal_match another WeightedTimeSeries with weight_size=" << like->weight_size << endl;

    internal_match_weights (like);
  }
}

void dsp::WeightedTimeSeries::internal_match_weights (const WeightedTimeSeries* like)
{
  uint64_t required = like->weight_size;

  if (weight_size < required)
  {
    if (base)
    {
      if (verbose)
        cerr << "dsp::WeightedTimeSeries::internal_match_weights Memory::free" << endl;
      weights_memory->do_free(base); base = weights = 0;
    }

    if (verbose)
      cerr << "dsp::WeightedTimeSeries::internal_match_weights Memory::allocate (" << required << ")" << endl;

    base = reinterpret_cast<uint16_t*> (weights_memory->do_allocate(required*sizeof(uint16_t)));
    if (!base)
      throw Error (InvalidState,"dsp::WeightedTimeSeries::internal_match_weights",
                  "could not allocate " UI64 " bytes", required);

    weight_size = required;
  }

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::internal_match_weights this->weight_size=" << weight_size
         << " that->weight_size=" << required << endl;

  weight_subsize = like->weight_subsize;
  weight_idat = like->weight_idat;

  unsigned reserve_weights = like->weights - like->base;

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::internal_match_weights reserve_weights=" << reserve_weights << endl;

  weights = base + reserve_weights;
}

void dsp::WeightedTimeSeries::set_reserve_kludge_factor (unsigned factor)
{
  reserve_kludge_factor = factor;
}

//! Offset the base pointer by offset time samples
void dsp::WeightedTimeSeries::seek (int64_t offset)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::seek (" << offset << ") "
      "base=" << base << " weights=" << weights << " diff=" << weights - base << endl;

  if (!offset)
    return;

  TimeSeries::seek (offset);

  if (!ndat_per_weight)
    return;

  offset += weight_idat;

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::seek weight_idat=" << weight_idat
	 << " new offset=" << offset << endl;

  if (offset > 0)
  {
    weights += offset / ndat_per_weight;
    weight_idat = offset % ndat_per_weight;
  }
  else if (offset < 0)
  {
    uint64_t back = -offset;
    uint64_t wback = back/ndat_per_weight;
    uint64_t wleft = back%ndat_per_weight;

    if (!wleft)
      weight_idat = 0;
    else {
      wback ++;
      weight_idat = ndat_per_weight - wleft;
    }
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::seek wback=" << wback
	   << " new weight_idat=" << weight_idat << endl;

    weights -= wback;
  }

  assert (weights >= base);
  assert (weight_idat < ndat_per_weight);
}

//! Return pointer to the specified data block
uint16_t* dsp::WeightedTimeSeries::get_weights (unsigned ichan, unsigned ipol)
{
  if (!ndat_per_weight)
  {
    throw Error (InvalidState, "dsp::WeightedTimeSeries::get_weights", "ndat_per_weight=0");
  }
  return weights + (ichan * npol_weight + ipol) * weight_subsize;
}

//! Return pointer to the specified data block
const uint16_t*
dsp::WeightedTimeSeries::get_weights (unsigned ichan, unsigned ipol) const
{
  if (!ndat_per_weight)
  {
    throw Error (InvalidState, "dsp::WeightedTimeSeries::get_weights",
                 "ndat_per_weight=%u", ndat_per_weight);
  }
  return weights + (ichan * npol_weight + ipol) * weight_subsize;
}


const dsp::WeightedTimeSeries&
dsp::WeightedTimeSeries::operator = (const WeightedTimeSeries& copy)
{
  copy_configuration (&copy);
  TimeSeries::operator = (copy);
  return *this;
}

void dsp::WeightedTimeSeries::prepend_checks (const TimeSeries* pre, uint64_t pre_ndat)
{
  TimeSeries::prepend_checks (pre, pre_ndat);

#if 0

  /* This does not work in multi-threaded applications */

  const WeightedTimeSeries* wt = dynamic_cast<const WeightedTimeSeries*> (pre);
  if (!wt)
    throw Error (InvalidState, "dsp::WeightedTimeSeries::prepend_checks",
                 "pre is not a WeightedTimeSeries");

  uint64_t pwdat = (wt->weight_idat+pre_ndat)%ndat_per_weight;
  uint64_t twdat = (weight_idat)%ndat_per_weight;

  if (pwdat != twdat && pwdat != (twdat+1)%ndat_per_weight)
    throw Error (InvalidState, "dsp::WeightedTimeSeries::prepend_checks",
                 "pre->weight_idat=%u + pre_ndat="I64
                 " modulo ndat_per_weight=%u equals "I64
                 " is not contiguous with weight_idat=%u",
                 wt->weight_idat, pre_ndat,
                 ndat_per_weight, pwdat,
                 weight_idat);
#endif
}

void dsp::WeightedTimeSeries::copy_weights (const WeightedTimeSeries* copy, uint64_t idat_start, uint64_t copy_ndat)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_weights this->ndat_per_weight=" << ndat_per_weight << " that->ndat_per_weight=" << copy->ndat_per_weight << endl;

  if (!copy)
    return;

  if (ndat_per_weight==0 || copy->get_ndat_per_weight()==0)
    return;

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_weights idat_start=" << idat_start
         << " weight_idat this=" << weight_idat
         << " copy=" << copy->weight_idat << endl;

  uint64_t iwt_start = (idat_start + copy->weight_idat) / ndat_per_weight;
  weight_idat = (idat_start + copy->weight_idat) % ndat_per_weight;

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_weights iwt_start=" << iwt_start
         << " weight_idat=" << weight_idat << " copy_ndat=" << copy_ndat << endl;

  if (!copy_ndat)
    return;

  uint64_t copy_nweights = copy->have_nweights();

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_weights"
         << " copy_nweights=" << copy_nweights << endl;

  if (!copy_nweights)
    return;

  uint64_t nweights = get_nweights (weight_idat + copy_ndat);

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_weights"
         << " nweights=" << nweights << endl;

  uint64_t have_weights = have_nweights();

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_weights"
         << " have_nweights=" << have_weights
         << " copy->have_nweights=" << copy->have_nweights() << endl;

  if (copy_nweights < iwt_start + nweights)
    throw Error (InvalidState, "dsp::WeightedTimeSeries::copy_weights",
                 "copy_nweights=%u < iwt_start=%u + nweights=%u",
                 copy_nweights, iwt_start, nweights);

  if (nweights > have_weights)
    throw Error (InvalidState, "dsp::WeightedTimeSeries::copy_weights",
                 "nweights need=%u > have=%u" "\n\t"
                 "weight_idat this=%u copy=%u" "\n\t"
                 "copy_ndat=%u ppweight=%u",
                 nweights, have_weights, weight_idat, copy->weight_idat,
                 copy_ndat, ndat_per_weight);

  unsigned bad_weight_count = 0;
  unsigned total_weight_count = 0;

  if (weights_on_host && copy->weights_on_host)
  {
    for (unsigned ichan=0; ichan<get_nchan_weight(); ichan++)
    {
      for (unsigned ipol=0; ipol<get_npol_weight(); ipol++)
      {
        auto data1 = get_weights (ichan, ipol);
        DEBUG("COPY base=" << data1);
        auto data2 = copy->get_weights (ichan, ipol) + iwt_start;

        for (uint64_t iwt=0; iwt<nweights; iwt++)
        {
          DEBUG("COPY iwt=" << iwt << " weight=" << data2[iwt]);
          data1[iwt] = data2[iwt];

          if (data1[iwt] == 0)
            bad_weight_count ++;

          total_weight_count ++;
        }
      }
    }
  }
  else
  {
    throw Error (InvalidState, "dsp::WeightedTimeSeries::copy_weights", "only host-to-host copying of selected weights implemented");
  }

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_weights bad/total weights=" << bad_weight_count << "/" << total_weight_count << endl;
}

void dsp::WeightedTimeSeries::copy_all_weights (const WeightedTimeSeries* input)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_all_weights" << endl;

  this->internal_match_weights(input);

  void* dst = this->internal_get_weights_buffer();
  const void* src = input->internal_get_weights_buffer();
  size_t count = input->internal_get_weights_size();

  assert (this->internal_get_weights_size() >= count);

  if (!count)
    return;

#if HAVE_CUDA
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::copy_all_weights HAVE_CUDA" << endl;

  auto input_device_memory = dynamic_cast<const CUDA::DeviceMemory*>(input->get_weights_memory());
  auto output_device_memory = dynamic_cast<CUDA::DeviceMemory*>(this->get_weights_memory());

  // determine the kind of copy to perform
  cudaMemcpyKind kind;

  if (input_device_memory)
  {
    if (output_device_memory)
      kind = cudaMemcpyDeviceToDevice;
    else
      kind = cudaMemcpyDeviceToHost;
  }
  else
  {
    if (output_device_memory)
      kind = cudaMemcpyHostToDevice;
    else
      kind = cudaMemcpyHostToHost;
  }

  // determine the stream to use for asynchronous copy, if any
  cudaStream_t input_stream = 0;
  cudaStream_t output_stream = 0;
  cudaStream_t use_stream = 0;

  if (input_device_memory)
  {
    input_stream = input_device_memory->get_stream();
    use_stream = input_stream;
  }

  if (output_device_memory)
  {
    output_stream = output_device_memory->get_stream();
    use_stream = output_stream;
  }

  if (input_device_memory && output_device_memory && input_stream != output_stream)
  {
    // two CUDA devices are operating on different streams
    // synchronize both streams to ensure that no threads are using the memory on either stream
    cudaStreamSynchronize(input_stream);
    cudaStreamSynchronize(output_stream);
  }

  //! Performs copy and any required CPU/GPU synchronization
  TransferCUDA::copy (dst, src, count, kind, use_stream);

#else

  // both input and output weights are assumed to be on the same device
  /* (currently, only CPU/host memory is possible at this point in the code,
     but there could be another device in the future that knows how to copy.) */
  weights_memory->do_copy(dst, src, count);

#endif
}

//! Add each value in data to this
void dsp::WeightedTimeSeries::add (const TimeSeries* that)
{
  auto wts = dynamic_cast<const WeightedTimeSeries*> (that);

  if (wts)
    add (wts);
  else
    TimeSeries::add(that);
}

void dsp::WeightedTimeSeries::add (const WeightedTimeSeries* that)
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::operator add" << endl;

  TimeSeries::add(that);

  uint64_t nweights = get_nweights ();

  for (unsigned ichan=0; ichan<get_nchan_weight(); ichan++)
  {
    for (unsigned ipol=0; ipol<get_npol_weight(); ipol++)
    {
      auto data1 = get_weights (ichan, ipol);
      auto data2 = that->get_weights (ichan, ipol);

      for (uint64_t iwt=0; iwt<nweights; iwt++)
      {
        if (data1[iwt] == 0 || data2[iwt] == 0)
          data1[iwt] = 0;
        else
          data1[iwt] = (data1[iwt] + data2[iwt])/2;
      }
    }
  }
  check_weights ();
}

void dsp::WeightedTimeSeries::zero ()
{
  TimeSeries::zero ();
  neutral_weights ();
}

//! Check that each floating point value is zeroed if weight is zero
void dsp::WeightedTimeSeries::check_weights ()
{
  mask_weights ();
}

//! Set all weights to one
void dsp::WeightedTimeSeries::neutral_weights ()
{
  if (verbose)
    cerr << "dsp::WeightedTimeSeries::neutral_weights " << weight_size
	 << " weights" << endl;

  for (uint64_t i=0; i<weight_size; i++)
    base[i] = 1;

  weight_idat = 0;
}

uint64_t dsp::WeightedTimeSeries::get_nzero () const try
{
  uint64_t nweights = get_nweights ();
  uint64_t zeroes = 0;
  for (uint64_t i=0; i<nweights; i++)
  {
    if (weights[i] == 0)
    {
      DEBUG("dsp::WeightedTimeSeries::get_nzero ZERO[" << i << "]");
      zeroes ++;
    }
  }

  return zeroes;
}
catch (Error& error)
{
  throw error += "dsp::WeightedTimeSeries::get_nzero";
}

/*! The algorithm is written so that only two sections of the array are
  used at one time.  Should minimize the number of cache hits */
void dsp::WeightedTimeSeries::mask_weights () try
{
  uint64_t nweights = get_nweights ();
  unsigned nparts = get_npol_weight() * get_nchan_weight();

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::mask_weights nweights=" << nweights << " nparts=" << nparts << endl;

  // collect all of the bad weights in the first array
  for (unsigned ipart=1; ipart<nparts; ipart++)
  {
    auto wptr = weights + ipart * weight_subsize;
    for (unsigned iwt=0; iwt < nweights; iwt++)
    {
      if (wptr[iwt] == 0)
      {
        DEBUG("MASK1 " << iwt);
        weights[iwt] = 0;
      }
    }
  }

  // distribute the bad weights to all of the arrays
  for (unsigned ipart=1; ipart<nparts; ipart++)
  {
    auto wptr = weights + ipart * weight_subsize;
    for (unsigned iwt=0; iwt < nweights; iwt++)
    {
      if (weights[iwt] == 0)
      {
        DEBUG("MASK2 " << iwt);
        wptr[iwt] = 0;
      }
    }
  }
}
catch (Error& error)
{
  throw error += "dsp::WeightedTimeSeries::mask_weights";
}

void
dsp::WeightedTimeSeries::convolve_weights (unsigned nfft, unsigned nkeep) try
{
  if (ndat_per_weight >= nfft)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::convolve_weights ndat_per_weight="
           << ndat_per_weight << " >= nfft=" << nfft << endl;

    // the fft happens within one weight, no "convolution" required
    return;
  }

  if (get_ndat() + nkeep < nfft)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::convolve_weights ndat=" << get_ndat()
           << " + nkeep=" << nkeep << " is less than nfft=" << nfft << endl;

    return;
  }

  uint64_t nweights_tot = get_nweights();

  if (verbose)
  {
    uint64_t nzero = get_nzero ();
    cerr << "dsp::WeightedTimeSeries::convolve_weights nfft=" << nfft
	 << " nkeep=" << nkeep << "\n  ndat_per_weight=" << ndat_per_weight
	 << " nweights=" << nweights_tot << " weight_idat=" << weight_idat
         << " bad=" << nzero << endl;
  }

  if (ndat_per_weight == 0)
    return;

  double weights_per_dat = 1.0 / ndat_per_weight;

  uint64_t blocks = (get_ndat()+nkeep-nfft) / nkeep;
  uint64_t end_idat = blocks * nkeep;

  uint64_t total_bad = 0;
  uint64_t zero_start = 0;
  uint64_t zero_end = 0;

  for (uint64_t start_idat=0; start_idat < end_idat; start_idat += nkeep)
  {
    uint64_t wt_idat = start_idat + weight_idat;

    uint64_t start_weight = uint64_t(wt_idat * weights_per_dat);
    uint64_t end_weight = uint64_t(ceil ((wt_idat+nfft) * weights_per_dat));

    if (end_weight > nweights_tot)
      throw Error (InvalidState, "dsp::WeightedTimeSeries::convolve_weights",
		   "end_weight=%d > nweights=%d \n\t"
                   "blocks=" UI64 " end_idat=" UI64,
                   end_weight, nweights_tot, blocks, end_idat);

    if (verbose)
      cerr << "dsp::WeightedTimeSeries::convolve_weights start_weight="
	   << start_weight << " end_weight=" << end_weight << endl;

    uint64_t zero_weights = 0;
    uint64_t iweight = 0;
    for (iweight=start_weight; iweight<end_weight; iweight++)
      if (weights[iweight] == 0)
      {
        DEBUG("FOUND BAD " << iweight);
        zero_weights ++;
      }

    /* If there exists bad data in the transform, the whole transform
       must be flagged as invalid; otherwise, the FFT will mix the bad
       data into the good data.  However, we cannot immediately flag
       the current transform, as this will affect the next test.
       Therefore, flag only the previous data set and set the flag for
       the next loop. */

    for (iweight=zero_start; iweight < zero_end; iweight++)
    {
      DEBUG("inside FLAGGING " << iweight);
      weights[iweight] = 0;
      total_bad ++;
    }

    if (zero_weights == 0)
      zero_start = zero_end = 0;

    else
    {
      if (verbose)
        cerr << "dsp::WeightedTimeSeries::convolve_weights bad weights="
	     << zero_weights << endl;

      zero_start = start_weight;
      zero_end = uint64_t( ceil((start_idat+nkeep) * weights_per_dat) );

      if (verbose)
        cerr << "dsp::WeightedTimeSeries::convolve_weights"
                " flagging " << zero_end-zero_start << " bad weights" << endl;
    }
  }

  for (uint64_t iweight=zero_start; iweight < zero_end; iweight++)
  {
    DEBUG("outside FLAGGING " << iweight);
    weights[iweight] = 0;
    total_bad ++;
  }

  if (verbose)
  {
    uint64_t nzero = get_nzero ();
    cerr << "dsp::WeightedTimeSeries::convolve_weights bad=" << nzero <<
      "/" << nweights_tot << endl;
  }
}
catch (Error& error)
{
  throw error += "dsp::WeightedTimeSeries::convolve_weights";
}

void dsp::WeightedTimeSeries::scrunch_weights (unsigned nscrunch) try
{
  uint64_t nweights_tot = get_nweights();

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::scrunch_weights nscrunch=" << nscrunch
	 << " ndat_per_weight=" << ndat_per_weight
	 << " nweights=" << nweights_tot
         << " weight_idat=" << weight_idat << endl;

  if (!ndat_per_weight)
    return;

  // the points per weight after time resolution decreases
  double points_per_weight = double(ndat_per_weight) / double(nscrunch);

  if (points_per_weight >= 1.0)
  {
    if (verbose)
      cerr << "dsp::WeightedTimeSeries::scrunch_weights new points_per_weight="
	   << points_per_weight << endl;

    ndat_per_weight = unsigned (points_per_weight);

    bool leftover = (weight_idat % ndat_per_weight != 0);
    weight_idat /= ndat_per_weight;
    if (leftover)
      weight_idat ++;

    if (verbose)
      cerr << "dsp::WeightedTimeSeries::scrunch_weights new weight_idat="
           << weight_idat << endl;

    return;
  }

  if (verbose)
    cerr << "dsp::WeightedTimeSeries::scrunch_weights"
      " scrunching to 1 point per wieght" << endl;

  // reduce the number of weights by scrunching
  uint64_t new_nweights = nweights_tot / nscrunch;
  uint64_t extra = nweights_tot % nscrunch;

  if (extra)
    new_nweights ++;

  for (uint64_t iwt=0; iwt < new_nweights; iwt++)
  {
    auto indi_weight = weights + iwt * nscrunch;

    if ((iwt+1)*nscrunch > nweights_tot)
      nscrunch = unsigned(extra);

    for (unsigned ivt=0; ivt < nscrunch; ivt++) {
      if (*indi_weight == 0) {
        weights[iwt] = 0;
        break;
      }
      else if (ivt == 0)
        weights[iwt] = *indi_weight;
      else
        weights[iwt] += *indi_weight;

      indi_weight ++;
    }

    weights[iwt] /= nscrunch;
  }

  ndat_per_weight = 1;

}
catch (Error& error)
{
  throw error += "dsp::WeightedTimeSeries::scrunch_weights";
}

//! Set the weights memory manager
/*! Weights may be stored on a device other than the one used to store floating-point data. */
void dsp::WeightedTimeSeries::set_weights_memory (Memory* memory)
{
  weights_memory = memory;

  weights_on_host = dynamic_cast<MemoryHost*>(memory) != nullptr;
}

dsp::Memory* dsp::WeightedTimeSeries::get_weights_memory ()
{
  return weights_memory;
}

const dsp::Memory* dsp::WeightedTimeSeries::get_weights_memory () const
{
  return weights_memory;
}
