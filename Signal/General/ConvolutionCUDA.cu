//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 - 2025 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"

// #define _DEBUG 1
#include "debug.h"

#include "dsp/ConvolutionCUDA.h"
#include "CUFFTError.h"

#if HAVE_CUFFT_CALLBACKS
#include "dsp/ConvolutionCUDACallbacks.h"
#include <cufftXt.h>
#endif

#include <iostream>
#include <cassert>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

__global__ void k_multiply_conv (float2* d_fft, const __restrict__ float2 * kernel, unsigned npart)
{
  const unsigned npt = blockDim.x * gridDim.x;
  unsigned i = blockIdx.x*blockDim.x + threadIdx.x;

  // load the kernel for this fine channel
  const float2 k = kernel[i];

#if _DEBUG
  // if the index is more than blockDim.x beyond the end of the array, then too many blocks were launched
  unsigned limit = npt;
  if (i == limit)
  {
    printf("k_multiply_conv index=%u > limit=%u (npt=%u * npart=%u + blockDim.x=%u)\n", i, limit, npt, npart, blockDim.x);
  }
  assert(i < limit);

  if (i == 0)
  {
    printf("k_multiply_conv blockIdx.y=%u npt=%u\n", blockIdx.y, npt);
  }
#endif

  d_fft += blockIdx.y * npt;
  if (i < npt)
  {
    d_fft[i] = cuCmulf(d_fft[i], k);
  }
}

__global__ void k_ncopy_conv (float2* output_data, unsigned output_stride,
           const float2* input_data, unsigned input_stride,
           unsigned to_copy)
{
  // shift the input forward FFT by the required number of batches
  input_data += blockIdx.y * input_stride;

  // shift in output forward
  output_data += blockIdx.y * output_stride;

  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

#if _DEBUG
  // if the index is more than blockDim.x beyond the end of the array, then too many blocks were launched
  unsigned limit = to_copy + blockDim.x;
  if (index == limit)
  {
    printf("k_ncopy_conv index=%u > limit=%u (to_copy=%u + blockDim.x=%u)\n", index, limit, to_copy, blockDim.x);
  }
  assert(index < limit);

  if (index == 0)
  {
    printf("k_ncopy_conv blockIdx.y=%u input_stride=%u output_stride=%u\n", blockIdx.y, input_stride, output_stride);
  }
#endif

  if (index < to_copy)
    output_data[index] = input_data[index];
}

CUDA::ConvolutionEngine::ConvolutionEngine (cudaStream_t _stream)
{
  stream = _stream;

  // create plan handles
  cufftResult result;

  if (dsp::Operation::verbose)
  {
    int device = -1;
    cudaGetDevice(&device);
    cerr << "CUDA::ConvolutionEngine ctor stream=" << _stream << " device=" << device << endl;
  }

  result = cufftCreate (&plan_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::ConvolutionEngine",
                      "cufftCreate(plan_fwd)");

  result = cufftCreate (&plan_fwd_batched);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::ConvolutionEngine",
                      "cufftCreate(plan_fwd_batched)");

  result = cufftCreate (&plan_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::ConvolutionEngine",
                      "cufftCreate(plan_bwd)");

  result = cufftCreate (&plan_bwd_batched);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::ConvolutionEngine",
                      "cufftCreate(plan_bwd_batched)");

  nbatch = 0;
  npt_fwd = 0;
  npt_bwd = 0;

  work_area = 0;
  work_area_size = 0;

  buf = 0;
  buf_zdm = 0;
  d_kernels = 0;
}

CUDA::ConvolutionEngine::~ConvolutionEngine()
{
  cufftResult result;

  result = cufftDestroy (plan_fwd);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::ConvolutionEngine::~ConvolutionEngine cufftDestroy(plan_fwd) failed" << endl;

  result = cufftDestroy (plan_fwd_batched);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::ConvolutionEngine::~ConvolutionEngine cufftDestroy(plan_fwd_batched) failed" << endl;

  result = cufftDestroy (plan_bwd);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::ConvolutionEngine::~ConvolutionEngine cufftDestroy(plan_bwd) failed" << endl;

  result = cufftDestroy (plan_bwd_batched);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::ConvolutionEngine::~ConvolutionEngine cufftDestroy(plan_bwd_batched)" << endl;

  cudaError_t error = cudaFree (work_area);
  if (error != cudaSuccess)
    cerr << "CUDA::ConvolutionEngine::~ConvolutionEngine cudaFree(work_area) failed" << endl;

  error = cudaFree (buf_zdm);
  if (error != cudaSuccess)
    cerr << "CUDA::ConvolutionEngine::~ConvolutionEngine cudaFree(buf_zdm) failed" << endl;

  error = cudaFree (buf);
  if (error != cudaSuccess)
    cerr << "CUDA::ConvolutionEngine::~ConvolutionEngine cudaFree(buf) failed" << endl;

  error = cudaFree (d_kernels);
  if (error != cudaSuccess)
    cerr << "CUDA::ConvolutionEngine::~ConvolutionEngine cudaFree(d_kernels) failed" << endl;
}

void CUDA::ConvolutionEngine::set_scratch (void * scratch)
{
  d_scratch = (cufftComplex *) scratch;
}

// prepare all relevant attributes for the engine
void CUDA::ConvolutionEngine::prepare (dsp::Convolution * convolution)
{
  const dsp::Response* response = convolution->get_response();

  npt_bwd = response->get_ndat();
  npt_fwd = convolution->get_minimum_samples();
  nsamp_overlap = convolution->get_minimum_samples_lost();
  nsamp_step = npt_fwd - nsamp_overlap;
  nfilt_pos = response->get_impulse_pos ();
  nfilt_neg = response->get_impulse_neg ();

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::prepare npt_fwd=" << npt_fwd << " npt_bwd=" << npt_bwd << endl;

  if (convolution->get_input()->get_state() == Signal::Nyquist)
    type_fwd = CUFFT_R2C;
  else
    type_fwd = CUFFT_C2C;

  // configure the singular FFT
  setup_singular ();

  unsigned npart = maximum_batched_nfft / npt_fwd;

  if (npart > 1)
    setup_batched (npart);
  else
    nbatch = 0;

#if HAVE_CUFFT_CALLBACKS
  setup_callbacks_ConvolutionCUDA (plan_fwd, plan_bwd, plan_fwd_batched, plan_bwd_batched, d_kernels, nbatch, stream);
#endif

  // initialize the kernel size configuration
  launch_config.init();
  launch_config.set_nelement (npt_bwd);
}

//! Copy the response from CPU to GPU memory
void CUDA::ConvolutionEngine::copy_response (const dsp::Response* response)
{
  // copy all kernels from host to device
  const float* kernel = response->get_datptr (0,0);

  unsigned nchan = response->get_nchan();
  unsigned ndat = response->get_ndat();

  unsigned need_size = ndat * sizeof(cufftComplex) * nchan;

  cudaError_t error;

  // allocate memory for dedispersion kernel of all channels
  if (kernels_size < need_size)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::ConvolutionEngine::copy_response kernels_size=" << kernels_size << " need_size=" << need_size << endl;
    
    if (d_kernels)
    {
      cudaFree(d_kernels);
      d_kernels = nullptr;
    }

    error = cudaMalloc ((void**)&d_kernels, need_size);

    if (error != cudaSuccess)
    {
      throw Error (InvalidState, "CUDA::ConvolutionEngine::copy_response",
        "could not allocate device memory for dedispersion kernel");
    }

    kernels_size = need_size;
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::copy_response cudaMemcpy stream=" << stream
         << " size=" << kernels_size << " d_kernel=" << d_kernels << endl;

  if (stream)
    error = cudaMemcpyAsync (d_kernels, kernel, kernels_size, cudaMemcpyHostToDevice, stream);
  else
    error = cudaMemcpy (d_kernels, kernel, kernels_size, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngine::copy_response",
     "could not copy dedispersion kernel to device");
  }

#if HAVE_CUFFT_CALLBACKS
  error = cudaMallocHost ((void **) h_conv_params, sizeof(unsigned) * 5);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::ConvolutionEngine::copy_response",
                 "could not allocate memory for h_conv_params");

  h_conv_params[0] = 0;
  h_conv_params[1] = npt_bwd;
  h_conv_params[2] = nfilt_pos;
  h_conv_params[3] = npt_bwd - nfilt_neg;
  h_conv_params[4] = nfilt_pos + nfilt_neg;

  setup_callbacks_conv_params (h_conv_params, sizeof(h_conv_params), stream);

#endif
}

void CUDA::ConvolutionEngine::setup_singular ()
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::setup_singular fwd=" << npt_fwd
         << " bwd=" << npt_bwd << endl;

  // setup forward plan
  cufftResult result = cufftPlan1d (&plan_fwd, npt_fwd, type_fwd, 1);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_singular",
                      "cufftPlan1d(plan_fwd)");

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::setup_singular cufftSetStream stream=" << stream << endl;

  result = cufftSetStream (plan_fwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_singular",
          "cufftSetStream(plan_fwd)");

  // setup backward plan
  result = cufftPlan1d (&plan_bwd, npt_bwd, CUFFT_C2C, 1);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_singular",
                      "cufftPlan1d(plan_bwd)");

  result = cufftSetStream (plan_bwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_singular",
                      "cufftSetStream(plan_bwd)");

  size_t buffer_size = npt_bwd * sizeof (cufftComplex);

  // R2C FFT of n real values yields n/2+1 complex value
  // https://docs.nvidia.com/cuda/cufft/index.html#data-layout
  if (type_fwd == CUFFT_R2C)
  {
    buffer_size += sizeof (cufftComplex);
  }

  cudaError_t error = cudaMalloc ((void **) &buf, buffer_size);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_singular",
                 "cudaMalloc(%x, %u): %s", &buf, buffer_size,
                 cudaGetErrorString (error));

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::setup_singular buf=" << buf << " buffer_size=" << buffer_size << endl;

  // R2C convolutions requires a buffer for the zero DM output
  if (type_fwd == CUFFT_R2C)
  {
    cudaError_t error = cudaMalloc ((void **) &buf_zdm, buffer_size);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_singular",
                   "cudaMalloc(%x, %u): %s", &buf_zdm, buffer_size,
                   cudaGetErrorString (error));
  }
}


// configure the singular and batched FFT plans
void CUDA::ConvolutionEngine::setup_batched (unsigned _nbatch)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::setup_batched npt_fwd=" << npt_fwd << " nsamp_step=" << nsamp_step
         << " npt_bwd=" << npt_bwd << " nbatch=" << _nbatch << endl;

  nbatch = _nbatch;

  int rank = 1;

  // complex layout plans for input
  int inembed[1] = { npt_fwd };
  int onembed[1] = { npt_bwd };

  int istride = 1;
  int ostride = 1;

  // the input moves forward a shorter amount
  int idist = nsamp_step;
  int odist = npt_bwd;

  size_t work_size_fwd = 0;
  cufftResult result = cufftMakePlanMany (plan_fwd_batched, rank, &npt_fwd,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              type_fwd, nbatch, &work_size_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftMakePlanMany (plan_fwd_batched)");

  result = cufftSetStream (plan_fwd_batched, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
          "cufftSetStream(plan_fwd_batched)");

  // get a rough estimate on work buffer size
  work_size_fwd = 0;
  result = cufftEstimateMany(rank, &npt_fwd,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             type_fwd, nbatch, &work_size_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftEstimateMany(plan_fwd)");

  // complex layout plans for input
  inembed[0] = npt_bwd;
  onembed[0] = nsamp_step;

  istride = 1;
  ostride = 1;

  // the output moves forward a shorter amount
  idist = npt_bwd;
  odist = nsamp_step;

  size_t work_size_bwd = 0;
  // the backward FFT is a has a simple layout (npt_bwd)
  DEBUG("CUDA::ConvolutionEngine::setup_batched cufftMakePlanMany (plan_bwd_batched)");
  result = cufftMakePlanMany (plan_bwd_batched, rank, &npt_bwd,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, nbatch, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftMakePlanMany (plan_bwd_batched)");

  result = cufftSetStream (plan_bwd_batched, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftSetStream(plan_bwd_batched)");

  DEBUG("CUDA::ConvolutionEngine::setup_batched bwd FFT plan set");

  work_size_bwd = 0;
  result = cufftEstimateMany(rank, &npt_bwd,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2C, nbatch, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftEstimateMany(plan_fwd)");

  work_area_size = (work_size_fwd > work_size_bwd) ? work_size_fwd : work_size_bwd;
  int auto_allocate = work_area_size > 0;

  DEBUG("CUDA::ConvolutionEngine::setup_batched cufftSetAutoAllocation(plan_fwd)");
  result = cufftSetAutoAllocation(plan_fwd_batched, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftSetAutoAllocation(plan_bwd_batched, %d)",
                      auto_allocate);

  DEBUG("CUDA::ConvolutionEngine::setup_batched cufftSetAutoAllocation(plan_bwd_batched)");
  result = cufftSetAutoAllocation(plan_bwd_batched, auto_allocate);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_batched",
                      "cufftSetAutoAllocation(plan_bwd_batched, %d)", auto_allocate);

  // free the space allocated for buf in setup_singular
  cudaError_t error = cudaFree (buf);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_batched",
                 "cudaFree(%x): %s", &buf, cudaGetErrorString (error));

  size_t batched_buffer_size = npt_bwd * nbatch * sizeof (cufftComplex);
  error = cudaMalloc ((void **) &buf, batched_buffer_size);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_batched",
                 "cudaMalloc(%x, %u): %s", &buf, batched_buffer_size,
                 cudaGetErrorString (error));

  // R2C convolutions requires a buffer for the zero DM output
  if (type_fwd == CUFFT_R2C)
  {
    error = cudaFree (buf_zdm);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_batched",
                   "cudaFree(%x): %s", &buf_zdm, cudaGetErrorString (error));
    error = cudaMalloc ((void **) &buf_zdm, batched_buffer_size);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_batched",
                   "cudaMalloc(%x, %u): %s", &buf_zdm, batched_buffer_size,
                   cudaGetErrorString (error));
  }

  // allocate device memory for dedispsersion kernel (1 channel)

  if (work_area)
  {
    error = cudaFree (work_area);
    if (error != cudaSuccess)
        throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_batched",
                    "cudaFree(%xu): %s", &work_area,
                    cudaGetErrorString (error));
    work_area = 0;
  }

  if (work_area_size > 0)
  {
    DEBUG("CUDA::ConvolutionEngine::setup_batched cudaMalloc("<<work_area<<", "<<work_area_size<<")");
    error = cudaMalloc (&work_area, work_area_size);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_batched",
                   "cudaMalloc(%x, %u): %s", &work_area, work_area_size,
                   cudaGetErrorString (error));
  }
}

#if HAVE_CUFFT_CALLBACKS
/*
void CUDA::ConvolutionEngine::setup_callbacks ()
{
  cudaError_t error;
  cufftResult_t result;

  cufftCallbackStoreC h_store_fwd;
  cufftCallbackStoreC h_store_bwd;
  cufftCallbackStoreC h_store_fwd_batch;
  cufftCallbackStoreC h_store_bwd_batch;

  error = cudaMemcpyFromSymbolAsync(&h_store_fwd, d_store_fwd,
                                    sizeof(h_store_fwd), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_fwd");

  error = cudaMemcpyFromSymbolAsync(&h_store_bwd, d_store_bwd,
                                    sizeof(h_store_bwd), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_bwd");

  error = cudaMemcpyFromSymbolAsync(&h_store_fwd_batch, d_store_fwd_batch,
                                    sizeof(h_store_fwd_batch), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_fwd_batch");

  error = cudaMemcpyFromSymbolAsync(&h_store_bwd_batch, d_store_bwd_batch,
                                    sizeof(h_store_bwd_batch), 0,
                                    cudaMemcpyDeviceToHost, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::ConvolutionEngine::setup_callbacks",
                 "cudaMemcpyFromSymbolAsync failed for h_store_bwd_batch");

  result = cufftXtSetCallback (plan_fwd, (void **)&h_store_fwd,
                               CUFFT_CB_ST_COMPLEX, (void **)&d_kernels);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
      "cufftXtSetCallback (plan_fwd, h_store_fwd)");

  result = cufftXtSetCallback (plan_bwd, (void **)&h_store_bwd,
                               CUFFT_CB_ST_COMPLEX, 0);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
      "cufftXtSetCallback (plan_bwd, h_store_bwd)");

  if (nbatch > 0)
  {
    result = cufftXtSetCallback (plan_fwd_batched, (void **)&h_store_fwd_batch,
                                 CUFFT_CB_ST_COMPLEX, (void **)&d_kernels);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
        "cufftXtSetCallback (plan_fwd_batched, h_store_fwd_batch)");

    result = cufftXtSetCallback (plan_bwd_batched, (void **)&h_store_bwd_batch,
                                 CUFFT_CB_ST_COMPLEX, 0);
    if (result != CUFFT_SUCCESS)
      throw CUFFTError (result, "CUDA::ConvolutionEngine::setup_callbacks",
        "cufftXtSetCallback (plan_bwd_batched, h_store_bwd_batch)");
  }
}
*/
#endif

void CUDA::ConvolutionEngine::perform (
  const dsp::TimeSeries* input,
  dsp::TimeSeries* output,
  unsigned npart
)
{
  perform(input, output, NULL, npart);
}

// Perform convolution choosing the optimal batched size or if ndat is not as
// was configured, then perform singular
void CUDA::ConvolutionEngine::perform (
  const dsp::TimeSeries* input,
  dsp::TimeSeries* output,
  dsp::TimeSeries* output_zdm,
  unsigned npart
)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::perform (" << npart << ")" << endl;

  if (npart == 0)
    return;

  if (type_fwd == CUFFT_C2C)
    perform_complex (input, output, output_zdm, npart);
  else
    perform_real (input, output, output_zdm, npart);
}

void CUDA::ConvolutionEngine::perform_complex (
  const dsp::TimeSeries* input,
  dsp::TimeSeries * output,
  dsp::TimeSeries* output_zdm,
  unsigned npart
)
{
  assert(input != nullptr);
  assert(output != nullptr);

  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();

  cufftComplex * in = nullptr;
  cufftComplex * out = nullptr;
  cufftComplex * out_zdm = nullptr;
  cufftResult result;

  const unsigned in_step_batch  = nsamp_step * nbatch;
  const unsigned out_step_batch = nsamp_step * nbatch;

  unsigned nbp = 0;
  if (nbatch > 0)
    nbp = npart / nbatch;

  if (dsp::Operation::verbose)
  {
    int device = -1;
    cudaGetDevice(&device);

    cerr << "CUDA::ConvolutionEngine::perform_complex device=" << device << " stream=" << stream << endl;
    cerr << "CUDA::ConvolutionEngine::perform_complex npart=" << npart << " nbatch=" << nbatch
         << " npb=" << nbp << " nsamp_step=" << nsamp_step << endl;
    cerr << "CUDA::ConvolutionEngine::perform_complex launch_config.get_nthread=" << launch_config.get_nthread()
         << " launch_config.get_nblock=" << launch_config.get_nblock() << endl;
    cerr << "CUDA::ConvolutionEngine::perform_complex buf=" << buf << " d_kernel=" << d_kernels << endl;
    cerr << "CUDA::ConvolutionEngine::perform_complex in_step_batch=" << in_step_batch << " out_step_batch=" << out_step_batch << endl;
  }

#if !HAVE_CUFFT_CALLBACKS
  auto blocks = dim3 (nsamp_step/launch_config.get_nthread(), nbatch, 1);
  if (nsamp_step % launch_config.get_nthread())
    blocks.x++;

  if (dsp::Operation::verbose)
    std::cerr << "CUDA::ConvolutionEngine::perform_complex BLOCKS x=" << blocks.x << " y=" << blocks.y << std::endl;
#endif

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {

#if HAVE_CUFFT_CALLBACKS
    // determine convolution kernel offset
    h_conv_params[0] = ichan * npt_bwd;
    setup_callbacks_conv_params (h_conv_params, sizeof(unsigned), stream);
#else
    const unsigned k_offset = ichan * npt_bwd;
#endif

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      in  = (cufftComplex *) input->get_datptr (ichan, ipol);
      out = (cufftComplex *) output->get_datptr (ichan, ipol);

      if (output_zdm != NULL)
      {
        // simply copy from input to output, excluding nfilt_pos and nfilt_neg
        out_zdm = (cufftComplex *) output_zdm->get_datptr (ichan, ipol);

        unsigned nsamp_zdm = nsamp_step * npart;
        cudaMemcpyAsync (out_zdm,
                         in + nfilt_pos,
                         nsamp_zdm * sizeof(cufftComplex),
                         cudaMemcpyDeviceToDevice,
                         stream);
      }

      // for each batched FFT
      for (unsigned i=0; i<nbp; i++)
      {
        DEBUG("CUDA::ConvolutionEngine::perform_complex ichan=" << ichan << " ipol=" << ipol << " ibatch=" << i);

        // perform forward batched FFT
        result = cufftExecC2C (plan_fwd_batched, in, buf, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_fwd_batched)");

#if HAVE_CUFFT_CALLBACKS
        // perform the inverse batched FFT (out-of-place)
        result = cufftExecC2C (plan_bwd_batched, buf, out, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_bwd_batched)");

#else

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_complex after cufftExecC2C(plan_fwd_batched)", stream);

        DEBUG("CUDA::ConvolutionEngine::perform_complex buf=" << buf << " d_kernel=" << d_kernels << " k_offset=" << k_offset << " nbatch=" << nbatch);

        // multiply by the dedispersion kernel
        k_multiply_conv<<<blocks,launch_config.get_nthread(),0,stream>>> (buf, d_kernels + k_offset, nbatch);

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_complex after k_multiply_conv (batched)", stream);
        
        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd_batched, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_bwd_batched)");

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_complex after cufftExecC2C(plan_bwd_batched)", stream);

        // copy batches of output from input
        k_ncopy_conv<<<blocks,launch_config.get_nthread(),0,stream>>> (out, nsamp_step,
                                                       buf + nfilt_pos, npt_bwd,
                                                       out_step_batch);
        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_complex after k_ncopy_conv (batched)", stream);
#endif

        out += out_step_batch;
        in  += in_step_batch;
      }

      for (unsigned ipart=nbp*nbatch; ipart<npart; ipart++)
      {
        DEBUG("CUDA::ConvolutionEngine::perform_complex ichan=" << ichan << " ipol=" << ipol << " ipart=" << ipart);

        result = cufftExecC2C (plan_fwd, in, buf, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_fwd)");

#if HAVE_CUFFT_CALLBACKS
        result = cufftExecC2C (plan_bwd, buf, out, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_complex",
                            "cufftExecC2C(plan_bwd)");
#else

        DEBUG("CUDA::ConvolutionEngine::perform_complex buf=" << buf << " d_kernel=" << d_kernels << " k_offset=" << k_offset);

        // multiply by the dedispersion kernel
        k_multiply_conv<<<launch_config.get_nblock(),launch_config.get_nthread(),0,stream>>> (buf, d_kernels + k_offset, 1);

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_complex after k_multiply_conv", stream);

        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform",
                            "cufftExecC2C(plan_bwd)");

        // copy from buffer to output
        k_ncopy_conv<<<launch_config.get_nblock(),launch_config.get_nthread(),0,stream>>> (out, nsamp_step,
                                                         buf + nfilt_pos, npt_bwd,
                                                         nsamp_step);
#endif

        in  += nsamp_step;
        out += nsamp_step;

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_complex after k_ncopy_conv", stream);
      }
    }
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::ConvolutionEngine::perform_complex", stream);
}

void CUDA::ConvolutionEngine::perform_real(
  const dsp::TimeSeries* input,
  dsp::TimeSeries* output,
  dsp::TimeSeries* output_zdm,
  unsigned npart
)
{
  assert(input != nullptr);
  assert(output != nullptr);

  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();

  assert(output->get_nchan() == nchan);
  assert(output->get_npol() == npol);
  assert(output->get_ndim() == 2);
  
  cufftReal* in = nullptr;
  cufftComplex* out = nullptr;
  cufftComplex* out_zdm = nullptr;
  cufftResult result;

  const unsigned out_nsamp_step = nsamp_step / 2;

  const unsigned in_step_batch  = nsamp_step * nbatch;
  const unsigned out_step_batch = out_nsamp_step * nbatch;

  unsigned nbp = 0;
  if (nbatch > 0)
    nbp = npart / nbatch;

#if !HAVE_CUFFT_CALLBACKS
  auto blocks = dim3 (out_nsamp_step/launch_config.get_nthread(), nbatch, 1);
  if (out_nsamp_step % launch_config.get_nthread())
    blocks.x++;

  if (dsp::Operation::verbose)
    std::cerr << "CUDA::ConvolutionEngine::perform_real BLOCKS x=" << blocks.x << " y=" << blocks.y << std::endl;
#endif

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngine::perform_real nchan=" 
         << nchan << " npol=" << npol << " out_nsamp_step=" << out_nsamp_step << " npt_bwd=" << npt_bwd << " nblock=" << launch_config.get_nblock() << endl;

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    const unsigned k_offset = ichan * npt_bwd;

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      in  = (cufftReal *) input->get_datptr (ichan, ipol);
      out = (cufftComplex *) output->get_datptr (ichan, ipol);

      if (output_zdm)
        out_zdm = (cufftComplex *) output_zdm->get_datptr (ichan, ipol);

      // for each batched FFT
      for (unsigned i=0; i<nbp; i++)
      {
        // perform forward batched FFT
        result = cufftExecR2C (plan_fwd_batched, in, buf);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_real",
                            "cufftExecC2C(plan_fwd_batched)");

        // require a zero DM version of the output
        if (output_zdm != NULL)
        {
          // perform the inverse batched FFT (in-place)
          result = cufftExecC2C (plan_bwd_batched, buf, buf_zdm, CUFFT_INVERSE);
          if (result != CUFFT_SUCCESS)
            throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_real",
                              "cufftExecC2C(plan_bwd_batched)");

          // copy batches of output from input
          k_ncopy_conv<<<blocks,launch_config.get_nthread(),0,stream>>> (out_zdm, out_nsamp_step,
                                                         buf_zdm + nfilt_pos, npt_bwd,
                                                         out_step_batch);
        }

        // multiply by the dedispersion kernel
        k_multiply_conv<<<blocks,launch_config.get_nthread(),0,stream>>> (buf, d_kernels + k_offset, nbatch);

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_real after batched k_multiply_conv", stream);

        // perform the inverse batched FFT (in-place)
        result = cufftExecC2C (plan_bwd_batched, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_real",
                            "cufftExecC2C(plan_bwd_batched)");

        // copy batches of output from input
        k_ncopy_conv<<<blocks,launch_config.get_nthread(),0,stream>>> (out, out_nsamp_step,
                                                       buf + nfilt_pos, npt_bwd,
                                                       out_step_batch);

        in  += in_step_batch;
        out += out_step_batch;
      }

      for (unsigned ipart=nbp*nbatch; ipart<npart; ipart++)
      {
        result = cufftExecR2C (plan_fwd, in, buf);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_real",
                            "cufftExecC2C(plan_fwd)");

        // require a zero DM version of the output
        if (output_zdm != NULL)
        {
          // perform the inverse batched FFT (in-place)
          result = cufftExecC2C (plan_bwd, buf, buf_zdm, CUFFT_INVERSE);
          if (result != CUFFT_SUCCESS)
            throw CUFFTError (result, "CUDA::ConvolutionEngine::perform_real",
                              "cufftExecC2C(plan_bwd_batched)");

          // copy batches of output from input
          k_ncopy_conv<<<launch_config.get_nblock(),launch_config.get_nthread(),0,stream>>> (out_zdm, out_nsamp_step,
                                                                buf_zdm + nfilt_pos, npt_bwd,
                                                                out_step_batch);
        }

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_real before k_multiply_conv", stream);
        
        // multiply by the dedispersion kernel
        if (dsp::Operation::verbose)
          cerr << "CUDA::ConvolutionEngine::perform_real ipart=" << ipart 
               << " npt=" << launch_config.get_nblock()*launch_config.get_nthread()
               << " buf=" << buf << " d_kernels=" << d_kernels << " k_offset=" << k_offset << endl;

        k_multiply_conv<<<launch_config.get_nblock(),launch_config.get_nthread(),0,stream>>> (buf, d_kernels + k_offset, 1);

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_real after k_multiply_conv", stream);

        // perform the inverse FFT (in-place)
        result = cufftExecC2C (plan_bwd, buf, buf, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS)
          throw CUFFTError (result, "CUDA::ConvolutionEngine::perform",
                            "cufftExecC2C(plan_bwd_batched)");

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_real before k_ncopy_conv", stream);
        
        // multiply by the dedispersion kernel
        if (dsp::Operation::verbose)
          cerr << "CUDA::ConvolutionEngine::perform_real ipart=" << ipart 
               << " blocks.x=" << blocks.x << " nthread=" << launch_config.get_nthread()
               << " out=" << out << " out_nsamp_step=" << out_nsamp_step << " out_nsamp_step=" << out_nsamp_step << endl;

        // copy batches of output from input
        k_ncopy_conv<<<launch_config.get_nblock(),launch_config.get_nthread(),0,stream>>> (out, out_nsamp_step,
                                                         buf + nfilt_pos, npt_bwd,
                                                         out_nsamp_step);

        if (dsp::Operation::verbose)
          check_error_stream("CUDA::ConvolutionEngine::perform_real after k_ncopy_conv", stream);

        in  += nsamp_step;
        out += out_nsamp_step;
      }
    }
  }
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream("CUDA::ConvolutionEngine::perform_real", stream);
}
