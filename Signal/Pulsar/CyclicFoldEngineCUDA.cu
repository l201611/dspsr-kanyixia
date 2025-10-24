//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2012-2025 by Glenn Jones, Paul Demorest, and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// #define _DEBUG 1

#include "dsp/CyclicFoldEngineCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>
#include <fstream>

using namespace std;
using namespace dsp;

CUDA::CyclicFoldEngineCUDA::CyclicFoldEngineCUDA (cudaStream_t _stream)
{
  use_set_bins = true;

  // no data on either the host or device
  synchronized = true;

  stream = _stream;
}

CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA ()
{
  if (lagbinplan)
  {
    if (Operation::verbose)
      cerr << "CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA freeing lagbinplan" <<endl;
    delete [] lagbinplan;
    lagbinplan = nullptr;
  }
  if (d_binplan)
  {
    if (Operation::verbose)
      cerr << "CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA freeing d_binplan" <<endl;
    cudaFree(d_binplan);
  }
  if (d_lagdata)
  {
    if (Operation::verbose)
      cerr << "CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA freeing d_lagdata" <<endl;
    cudaFree(d_lagdata);
  }

  if (Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::~CyclicFoldEngineCUDA finished" <<endl;
}

void CUDA::CyclicFoldEngineCUDA::synch (dsp::PhaseSeries *out) try
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::synch this=" << this << " synchronised=" << synchronized <<  endl;

  if (synchronized)
    return;

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::CyclicFoldEngineCUDA::synch output=" << output << endl;
    cerr << "CUDA::CyclicFoldEngineCUDA::synch transferring lag data synch out=" << out <<" out.ndat_folded=" << out->get_ndat_folded()
        << " lagdata_size=" << lagdata_size << endl;
  }
  // transfer lag data from GPU

  cudaError error;
  if (stream)
    error = cudaMemcpyAsync (lagdata,d_lagdata,lagdata_size*sizeof(float),cudaMemcpyDeviceToHost,stream);
  else
    error = cudaMemcpy (lagdata,d_lagdata,lagdata_size*sizeof(float),cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::sync",
                 "cudaMemcpy%s %s",
                 stream?"Async":"", cudaGetErrorString (error));

  cudaDeviceSynchronize();

  // Call usual synch() to do transform
  dsp::CyclicFoldEngine::synch(out);

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::synch now synch'ed" << endl;

  synchronized = true;
}
catch (Error& error)
{
  throw error += "CUDA::CyclicFoldEngineCUDA::synch";
}

void CUDA::CyclicFoldEngineCUDA::set_ndat (uint64_t _ndat, uint64_t _idat_start)
{
  // ndat is idat_end - idat_start
  // binplan_size is _ndat
  setup();

  if (Operation::verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat ndat=" << _ndat << endl;

  current_turn = 0;
  last_ibin = 0;
  ndat_fold = _ndat;
  idat_start = _idat_start;

  if (Operation::verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat "
      << "nlag=" << nlag << " "
      << "nbin=" << nbin << " "
      << "npol=" << npol_out << " "
      << "nchan=" << nchan << endl;

  uint64_t _lagdata_size = nlag * nbin * npol_out * ndim * nchan;

  if (Operation::verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat lagdata_size=" << _lagdata_size << endl;

  if (_lagdata_size > lagdata_size) 
  {
    if (Operation::verbose)
      cerr << "dsp::CyclicFoldEngine::set_ndat alloc lagdata" << endl;

    if (lagdata) delete [] lagdata;
    lagdata_size = _lagdata_size;
    lagdata = new float [lagdata_size];

    if (d_lagdata) cudaFree(d_lagdata);
    cudaMalloc((void**)&d_lagdata, lagdata_size * sizeof(float));
    cudaMemset(d_lagdata, 0, lagdata_size * sizeof(float));
  }
}

void CUDA::CyclicFoldEngineCUDA::set_bin (uint64_t idat, double d_ibin, double bins_per_sample)
{
  if (Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::set_bin (singular) disabled" << endl;

  return;
}

uint64_t CUDA::CyclicFoldEngineCUDA::get_bin_hits (int ibin)
{
  int iturn = 0;
  int idx = 0;
  idx = iturn*nbin*nlag + ibin*nlag; // we want the zero lag hits
  uint64_t hits = 0;
  while (idx < binplan_size)
  {
    hits += lagbinplan[idx].hits;
    iturn += 1;
    idx = iturn*nbin*nlag + ibin*nlag; // we want the zero lag hits
  }

  DEBUG("CyclicFoldEngineCUDA::get_bin_hits ibin=" << ibin << " hits=" << hits);

  return hits;
}

/*
  The bin plan index = iturn*nbin*nlag + ibin*nlag + ilag
  Each entry includes:
  - offset, the starting data sample,
  - hits, the number of data samples to include in this lag/bin, and
  - ibin, the bin index

  There is one entry for every lag, every bin, and for all turns in this data block
*/
uint64_t CUDA::CyclicFoldEngineCUDA::set_bins (double phi, double phase_per_sample, uint64_t _ndat, uint64_t idat_start)
{
  if (Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::set_bins" << endl;

  phi = phi - floor(phi); // fractional phase at start
  double samples_per_bin = (1.0 / nbin) * (1.0 / phase_per_sample); // (1 turn / nbin bins) * (turns (phase) / sample) ^ -1
  double nturns = _ndat * phase_per_sample; // total number of turns represented by this block of data
  double startph = phi;  // starting fractional phase, the smallest valid phase
  double endph = startph + nturns; // final phase, the largest valid phase of any data point
  unsigned intnturns = ceil(nturns) + 1;  // total number of turns in the binplan. This could probably be safely just ceil(nturns) but add 1 to be sure.

  int _binplan_size = intnturns*nbin*nlag; // total number of entries in the bin plan.

  if (Operation::verbose)
  {
    cerr << "CUDA::CyclicFoldEngineCUDA::set_bins start ph:" << startph << " intnturns:" << intnturns << " _ndat:" << _ndat << " nlag:" << nlag
         << " phase per sample:" << phase_per_sample << " nturns:" << nturns << endl;
    cerr << "CUDA::CyclicFoldEngineCUDA::set_bins binplansize:" << binplan_size << "  _binplansize:" << _binplan_size << endl;
  }

  // allocate memory for the binplan
  if (_binplan_size > binplan_size)
  {
    if (Operation::verbose)
      cerr << "dsp::CyclicFoldEngine::set_bins alloc binplan" << endl;

    if (lagbinplan)
    {
      delete [] lagbinplan;
    }

    lagbinplan = new bin [_binplan_size];

    if (d_binplan)
    {
      auto error = cudaFree(d_binplan);
      if (error != cudaSuccess)
        throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::set_bins",
              "cudaFree %s %s", cudaGetErrorString (error));
    }

    uint64_t mem_size = _binplan_size * sizeof(bin);
    auto error = cudaMalloc ((void **)&(d_binplan),mem_size);
    if (error != cudaSuccess)
      throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::set_bins",
             "cudaMalloc new %s %s",cudaGetErrorString (error));

    if (Operation::verbose)
      cerr << "dsp::CyclicFoldEngine::set_bins d_binplan=" << (void*) d_binplan << endl;

    binplan_size = _binplan_size;
  }

  memset(lagbinplan, 0 , sizeof(bin)*binplan_size);  // all entries start out with zero hits, so any uninitialized portions will be ignored by the folding threads
  ndat_fold = _ndat;

#if _DEBUG
  ofstream outdat ("gpu_binplan.dat");
  outdat << "iturn ibin ilag offset" << endl;
#endif

  for (unsigned iturn=0; iturn < intnturns; iturn++)
  {
    for (unsigned ibin = 0; ibin < nbin; ibin++)
    {
      for (unsigned ilag=0; ilag < nlag; ilag++)
      {
        // minph is the starting phase of valid data for this lag/bin
        // maxph is the ending phase
        // thus we want to include all data points with phases in between minph and maxph

        /*
          COMPLEX PHASE CONVENTION: PART 0

          Compared to equation 5 of Demorest (2011), where C(phi,tau) = <x(t+tau/2) x*(t-tau/2)>,
          CyclicFoldEngine::fold computes C(phi,-tau) = C*(phi,tau).

          C(phi,-tau) = < {x(t-tau/2) x*(t+tau/2); phi(t)=phi} >

          Let t' = t - tau/2, and compute

          C(phi,-tau) = < {x(t') x*(t'+tau); phi(t'+tau/2)=phi} >

          Note that minph/phase_per_sample and maxph/phase_per_sample are use to compute sample index,
          which is proportional to time, and ibin and iturn are proportional to phase; inverting

          phi = phase_per_sample * (t' + tau/2)

          yields

          t' = phi/phase_per_sample - tau/2

          or

          minph = t'*phase_per_sample = phi - 0.5*(tau*phase_per_sample)
        */

        double minph = (ibin*1.0)/nbin + iturn - 0.5*(ilag*phase_per_sample);
        double maxph = (ibin+1.0)/nbin + iturn - 0.5*(ilag*phase_per_sample);

        // index of this binplan entry
        unsigned planidx = iturn*nbin*nlag + ibin*nlag + ilag;

        if ( maxph > endph )
        {
          maxph = endph; // keep maxph from going off the end of the data block. In theory we should really pull more data from the next block, but for now
                  // we just ignore correlations that span more than one data block
        }

        if ((minph > endph) || (maxph < minph))
        {
          // if the start of this lag/bin data is past the end of the data block (minph > endph), there is no valid data for this lag/bin
          // if maxph < minph, then it must be that minph > endph because the only way for this to happen would be if maxph were reassigned to endph in the previous clause.
          lagbinplan[planidx].offset = 0;
          lagbinplan[planidx].ibin = 0;
          lagbinplan[planidx].hits = 0;
          continue;
        }

        if (minph > startph)
        {
          // The basic case, the lag/bin data is fully within the data block, or goes right up to the end of the block (in which case maxph=endph)
          lagbinplan[planidx].offset = round((minph-startph)/phase_per_sample);
          lagbinplan[planidx].ibin = ibin;
          lagbinplan[planidx].hits = round((maxph-minph)/phase_per_sample);
        }
        else if (maxph > startph)
        {
          // In this case, the start of the lag/bin data precedes the first available data point, but there is still valid data from startph to maxph
//          cerr << "minph < startph " << minph << " < " << startph << endl;
          lagbinplan[planidx].offset = 0;
          lagbinplan[planidx].ibin = ibin;
          lagbinplan[planidx].hits = round((maxph-startph)/phase_per_sample);
        }
        else {
          // Finally, here minph <= startph and maxph <= startph, so the data needed fully precedes this data block.
//          cerr << "maxph < startph " << minph << " < " << startph << endl;
          lagbinplan[planidx].offset = 0;
          lagbinplan[planidx].ibin = 0;
          lagbinplan[planidx].hits = 0;
        }

        if (lagbinplan[planidx].hits > 0)
        {
          unsigned end_dat = lagbinplan[planidx].offset + lagbinplan[planidx].hits;

          // verify that offset and hits are valid
          if (lagbinplan[planidx].offset >= ndat_fold)
          {
            if (Operation::verbose)
              cerr << "dsp::CyclicFoldEngine::set_bins invalid offset=" << lagbinplan[planidx].offset
                   << " for iturn=" << iturn << " ibin=" << ibin << " ilag=" << ilag << " planidx=" << planidx << endl;
            lagbinplan[planidx].hits = 0;
          }
          else if (end_dat > ndat_fold)
          {
            if (Operation::verbose)
              cerr << "dsp::CyclicFoldEngine::set_bins offset+hits=" << end_dat
                   << " for iturn=" << iturn << " ibin=" << ibin << " ilag=" << ilag << " planidx=" << planidx << endl;
            unsigned new_hits = ndat_fold - lagbinplan[planidx].offset;
            if (Operation::verbose)
              cerr << "dsp::CyclicFoldEngine::set_bins old hits=" << lagbinplan[planidx].hits << " new hits=" << new_hits << endl;
            lagbinplan[planidx].hits = new_hits;
          }
        }

#if _DEBUG
        if (lagbinplan[planidx].hits > 0)
          outdat << iturn << " " << ibin << " " << ilag << " " << lagbinplan[planidx].offset << endl;
#endif

      }
    }
  }
  return ndat_fold;
}


void CUDA::CyclicFoldEngineCUDA::zero ()
{
  dsp::CyclicFoldEngine::zero();
  if (d_lagdata && lagdata_size > 0)
  {
    if (Operation::verbose)
      cerr << "CUDA::CyclicFoldEngineCUDA::zero: zeroing lagdata on gpu" << endl;
    if (stream)
      cudaMemsetAsync(d_lagdata, 0, lagdata_size * sizeof(float), stream);
    else
    cudaMemset(d_lagdata, 0, lagdata_size * sizeof(float));
  }
  else
  {
    if (Operation::verbose)
      cerr << "CUDA::CyclicFoldEngineCUDA::zero: not doing anything because d_lagdata=" << d_lagdata << " and lagdata_size=" << lagdata_size << endl;
  }
}

void CUDA::CyclicFoldEngineCUDA::send_binplan ()
{
  if (binplan_size == 0)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::CyclicFoldEngineCUDA::send_binplan binplan_size == 0 (do nothing)" << endl;
    return;
  }

  uint64_t mem_size = binplan_size * sizeof(bin);

  if (dsp::Operation::verbose)
    cerr << "CUDA::CyclicFoldEngineCUDA::send_binplan ndat=" << ndat_fold
         << " mem_size " << mem_size
         << " binplan_size=" << binplan_size
         << " nlag=" << nlag
         << " sizeof(bin)=" << sizeof(bin)
         << " current_turn=" << current_turn
         << endl;

  cudaError error;

  if (Operation::verbose)
  {
    cerr << "CUDA::CyclicFoldEngineCUDA::send_binplan copying: stream=" << stream << " d_binplan=" << d_binplan
         << " mem_size=" << mem_size << " lagbinplan=" << lagbinplan << endl;
  }

  if (stream)
    error = cudaMemcpyAsync (d_binplan,lagbinplan,mem_size,cudaMemcpyHostToDevice,stream);
  else
    error = cudaMemcpy (d_binplan,lagbinplan,mem_size,cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::send_binplan",
                 "cudaMemcpy%s %s",
                 stream?"Async":"", cudaGetErrorString (error));
}

// This function is never used. Lagdata is trasfered by the synch call
void CUDA::CyclicFoldEngineCUDA::get_lagdata ()
{
  if (Operation::verbose)
    cerr << "CyclicFoldEngineCUDA::get_lagdata" << endl;

  size_t lagdata_bytes = lagdata_size * sizeof(float);
  cudaError error;
  if (stream)
    error = cudaMemcpyAsync (lagdata, d_lagdata, lagdata_bytes, cudaMemcpyDeviceToHost, stream);
  else
    error = cudaMemcpy (lagdata, d_lagdata, lagdata_bytes, cudaMemcpyDeviceToHost);

  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::CyclicFoldEngineCUDA::get_lagdata",
                 "cudaMemcpy%s %s",
                 stream?"Async":"", cudaGetErrorString (error));
}

/*
 *  CUDA Kernels
 *
 */
// Since there is a maximum number of threads per block which may be less than the number of lags times number of pols,
// the ilag index is split into ilag = ilagb*nlaga + ilaga, where nlaga will be such that nlaga*npol = max_threads_per_block
// Each thread calculates the cyclic correlation for one lag for one bin for one input channel for one pol
// threadIdx.x -> ilaga    blockDim.x
// threadIdx.y -> pol
// threadIdx.z -> not used
// blockIdx.x -> ilagb
// blockIdx.y -> ibin
// blockIdx.z = ichan

// data are in FPT order, so chunks of time for a given pol and frequency
// in_span gives size of one time chunk for a given freq and pol in floats
__global__ void cycFoldIndPol (const float* in_base,
                unsigned in_span,
                float* out_base,
                unsigned binplan_size,
                unsigned nlag,
                CUDA::bin* binplan)
{
  unsigned ilaga = threadIdx.x;
  unsigned nlaga = blockDim.x;
  unsigned ilagb = blockIdx.x;
  unsigned ibin = blockIdx.y;
  unsigned ichan = blockIdx.z;
  unsigned ipol = threadIdx.y;
  unsigned npol = blockDim.y;
  unsigned nbin = gridDim.y;
  unsigned nchan = gridDim.z;
  unsigned ilag = ilagb*nlaga + ilaga;

  if (ilag >= nlag)
  {
    return;
  }

  unsigned planidx = nlag*ibin+ilag;
  const unsigned ndim = 2; // always complex data assumed

  if (planidx >= binplan_size)
  {
    return;
  }

  in_base += in_span * (ichan*npol + ipol);  //in_span is in units of float, so no need to mult by ndim

  out_base += ndim*(ibin*npol*nchan*nlag
    + ipol*nchan*nlag
    + ichan*nlag
    + ilag);

  unsigned bpstep = nlag*nbin; // step size to get to the next rotation for a given lag and bin in the binplan

  float2 total = make_float2(0.0,0.0);

  const float2* in_base_complex = reinterpret_cast<const float2*> (in_base);

  for (; planidx < binplan_size; planidx += bpstep)
  {
    const float2* a = in_base_complex + binplan[planidx].offset;
    const float2* b = a + ilag;

    for (unsigned i=0; i < binplan[planidx].hits; i++)
    {
      /*
        COMPLEX PHASE CONVENTION: PART 1

        Compared to equation 5 of Demorest (2011), where C(phi,tau) = <x(t+tau/2) x*(t-tau/2)>,
        the following lines compute C(phi,-tau) = C*(phi,tau).  This complex conjugation / delay negation
        also negates the frequency axis; however, CyclicFoldEngine::synch computes the inverse C->R FFT,
        which double negates (restores) the frequency axis, making the output of dspsr consistent with the 
        expectation that an inverse FFT along the radio frequency axis will return positive delays in the
        first half of the result.
      */

      // total += a * conj(b)
      total.x += a[i].x*b[i].x + a[i].y*b[i].y;
      total.y += a[i].y*b[i].x - a[i].x*b[i].y;
    }
  }

  out_base[0] += total.x;
  out_base[1] += total.y;
}

// Since there is a maximum number of threads per block which may be less than the number of lags times number of pols,
// the ilag index is split into ilag = ilagb*nlaga + ilaga, where nlaga will be such that nlaga*npol = max_threads_per_block
// Each thread calculates the cyclic correlation for one lag for one bin for one input channel for one pol
// threadIdx.x -> ilaga    blockDim.x
// threadIdx.y -> pol
// blockIdx.x -> ilagb
// blockIdx.y -> ibin
// This version gets passed ichan and nchan directly (it operates just on one channel) because early cuda could not handle 3dim thread grids
// data is in FPT order, so chunks of time for a given pol and frequency
// in_span gives size of one time chunk for a given freq and pol in floats
__global__ void cycFoldIndPolOneChan (const float* in_base,
                unsigned in_span,
                float* out_base,
                unsigned binplan_size,
                unsigned nlag,
                CUDA::bin* binplan,
                unsigned nchan,
                unsigned ichan)
{
  unsigned ilaga = threadIdx.x;
  unsigned nlaga = blockDim.x;
  unsigned ilagb = blockIdx.x;
  unsigned ibin = blockIdx.y;
  unsigned ipol = threadIdx.y;
  unsigned npol = blockDim.y;
  unsigned nbin = gridDim.y;
  unsigned ilag = ilagb*nlaga + ilaga;
  if (ilag >= nlag){
    return;
  }
  unsigned planidx = nlag*ibin+ilag;
  const unsigned ndim = 2; // always complex data assumed

  if (planidx >= binplan_size) {
    return;
  }

  in_base  += in_span  * (ichan*npol + ipol);  //in_span is in units of float, so no need to mult by ndim
//  out_base += out_span * (ichan*npol + ipol);
  out_base += ndim*(ibin*npol*nchan*nlag
    + ipol*nchan*nlag
    + ichan*nlag
    + ilag);

  unsigned bpstep = nlag*nbin; // step size to get to the next rotation for a given lag and bin in the binplan

  float2 total = make_float2(0.0,0.0);

  const float2* in_base_complex = reinterpret_cast<const float2*> (in_base);

  for (; planidx < binplan_size; planidx += bpstep)
  {
    const float2* a = in_base_complex + binplan[planidx].offset;
    const float2* b = a + ilag;

    for (unsigned i=0; i < binplan[planidx].hits; i++)
    {
      /* See note about "COMPLEX PHASE CONVENTION: PART 1" */

      // total += a * conj(b)
      total.x += a[i].x*b[i].x + a[i].y*b[i].y;
      total.y += a[i].y*b[i].x - a[i].x*b[i].y;
    }
  }

  out_base[0] += total.x;
  out_base[1] += total.y;
}

void check_error (const char*);


void CUDA::CyclicFoldEngineCUDA::fold ()
{
  // TODO state/etc checks
  if (Operation::verbose)
    cerr << "CyclicFoldEngineCUDA::fold" << endl;

  setup ();
  send_binplan ();
  const unsigned THREADS_PER_BLOCK = 1024;
  unsigned nlaga,nlagb;
  // if nlag*npol < THREADS_PER_BLOCK then nlaga = nlag, nlagb = 1
  // else nlaga = THREADS_PER_BLOCK/npol, nlagb = nlag/nlaga + 1
  if (nlag*npol > THREADS_PER_BLOCK) {
    nlaga = THREADS_PER_BLOCK/npol;
    nlagb = nlag/nlaga + 1;
  }
  else {
    nlagb = 1;
    nlaga = nlag;
  }

  dim3 blockDim (nlaga, npol, 1);
//  dim3 gridDim (nlagb, nbin, nchan);
  dim3 gridDim (nlagb, nbin, 1);
  if (Operation::verbose){
    cerr << "nlag=" << nlag << " binplan_size=" << binplan_size << " input_span=" << input_span  << " d_lagdata=" << d_lagdata << endl;
    cerr << "blockDim=" << blockDim.x << "," << blockDim.y << "," << blockDim.z << "," << endl;
    cerr << "gridDim="  << gridDim.x << "," << gridDim.y << "," << gridDim.z << "," << endl;
  }
  unsigned lagbinplan_size = binplan_size;

  for(unsigned ichan=0;ichan < nchan; ichan++){
    cycFoldIndPolOneChan <<<gridDim,blockDim,0,stream>>>(input,
          input_span,
          d_lagdata,
          lagbinplan_size,
          nlag,
          d_binplan,
          nchan,
          ichan);
  }
  // profile on the device is no longer synchronized with the one on the host
  synchronized = false;
  cudaDeviceSynchronize();

  if (Operation::verbose)
    cerr << "CyclicFoldEngineCUDA::fold finished, syncronized=false" << endl;

  //if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::CyclicFoldEngineCUDA::fold cuda error: ");
}

