/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_TransformationTestHelper_h
#define __dsp_TransformationTestHelper_h

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/Memory.h"

#if HAVE_CUDA
#include "dsp/TransferCUDATestHelper.h"
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>

#include <cstdint>
#include <cassert>

namespace dsp::test
{
/**
 * @brief Manages a Transformation, its input and output, and copies to/from GPU as needed
 *
 */
template<class In, class Out>
class TransformationTestHelper
{
  public:

    typedef Transformation<In,Out> Xform;

    //! Test helper states
    enum State { Torn_Down,    // Initial state, and state after calling tear_down
                 Set_Up,       // State after calling set_up
                 Initialized   // State after calling initialize_operation
               };

    //! Set true when test will be run on gpu
    void set_on_gpu (bool flag);

    //! Initializes resources (can be called during SetUp)
    /*! Constructs and configures the input and output containers */
    void set_up ();

    //! Releases resources (can be called during TearDown)
    /*! Destroys the input and output containers */
    void tear_down ();

    //! Initializes resources
    void initialize_operation (Xform*);

    //! Perform the operation, including any necessary data transfer
    /*! If on_gpu, performs host-to-device transfer of input and device-to-host transfer of output */
    void perform_operation (Xform*);

#ifdef HAVE_CUDA
    // adding a non-virtual method does not change the size of the class
    cudaStream_t& get_cuda_stream ();
#endif

  protected:

    //! The input to the Transformation
    Reference::To<In> input;

    //! The output of the Transformation
    Reference::To<Out> output;

    //! The input to the Transformation on the GPU
    Reference::To<In> device_input;

    //! The output of the Transformation on the GPU
    Reference::To<Out> device_output;

    //! device memory manager
    Reference::To<dsp::Memory> device_memory;

    //! Cast to a cudaStream_t in get_cuda_stream
    uint64_t cuda_stream = 0;

    //! Set true when test will run on GPU
    bool on_gpu = false;

    //! Output container memory should be configured to be on device
    /*! Applies only when on_gpu is true.  This is set to false when the GPU
        implementation uses its own on-device memory (e.g. Fold operations) */
    bool output_on_device = true;

    //! The current state of the test helper
    State state = Torn_Down;
};

template<class In, class Out>
void TransformationTestHelper<In,Out>::set_on_gpu (bool flag)
{
  if (state != Torn_Down)
    throw Error (InvalidState, "TransformationTestHelper<In,Out>::set_on_gpu",
                 "cannot change on_gpu state when already setup");
  on_gpu = flag;
}

template<class In, class Out>
void TransformationTestHelper<In,Out>::set_up ()
{
  if (state != Torn_Down)
    throw Error (InvalidState, "TransformationTestHelper<In,Out>::set_up",
                 "cannot setup when already setup");

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    cudaStreamCreate(&get_cuda_stream());
    device_input = new In;

    device_memory = new CUDA::DeviceMemory(get_cuda_stream());
    device_input->set_memory(device_memory);

    if (output_on_device)
    {
      device_output = new Out;
      device_output->set_memory(device_memory);
    }
  }
#endif

  input = new In;
  output = new Out;

  time_t temp = time(nullptr);
  struct tm date = *gmtime(&temp);
  input->set_start_time(MJD(date));
  input->set_rate(1.0);

  state = Set_Up;
}

template<class In, class Out>
void TransformationTestHelper<In,Out>::tear_down ()
{
  if (state == Torn_Down)
    throw Error (InvalidState, "TransformationTestHelper<In,Out>::tear_down",
                 "cannot teardown when not setup");

  input = nullptr;
  output = nullptr;
  device_input = nullptr;
  device_output = nullptr;
  device_memory = nullptr;

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    cudaStreamDestroy(get_cuda_stream());
  }
#endif

  state = Torn_Down;
}

//! Initialize the current test parameters
template<class In, class Out>
void TransformationTestHelper<In,Out>::initialize_operation (Xform* xform)
{
  if (Operation::verbose)
    std::cerr << "TransformationTestHelper<In,Out>::initialize_operation" << std::endl;

  if (state != Set_Up)
    throw Error (InvalidState, "TransformationTestHelper<In,Out>::initialize_operation",
                 "cannot initialize operation when not setup");

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    device_input->copy_configuration(input);
    xform->set_input(device_input);

    if (output_on_device)
      xform->set_output(device_output);
    else
      xform->set_output(output);
  }
  else
#endif
  {
    if (Operation::verbose)
      std::cerr << "TransformationTestHelper<In,Out>::initialize_operation setting input and output" << std::endl;
    xform->set_input(input);
    xform->set_output(output);
  }

  state = Initialized;
}

template<class In, class Out>
void TransformationTestHelper<In,Out>::perform_operation (Xform* xform)
{
  if (state != Initialized)
    throw Error (InvalidState, "TransformationTestHelper<In,Out>::perform_operation",
                 "cannot perform operation when operation is not initialized");

#ifdef HAVE_CUDA
  if (on_gpu)
  {
    if (dsp::Operation::verbose)
      std::cerr << "TransformationTestHelper<In,Out>::perform_operation copy host-to-device" << std::endl;
    TransferCUDATestHelper xfer;
    xfer.copy(device_input, input, cudaMemcpyHostToDevice);
  }
#endif

  xform->prepare();
  xform->operate();

#ifdef HAVE_CUDA
  if (on_gpu && output_on_device)
  {
    if (dsp::Operation::verbose)
      std::cerr << "TransformationTestHelper<In,Out>::perform_operation copy device-to-host" << std::endl;
    TransferCUDATestHelper xfer;
    xfer.copy(output, device_output, cudaMemcpyDeviceToHost);
  }
#endif
}

#ifdef HAVE_CUDA
template<class In, class Out>
cudaStream_t& TransformationTestHelper<In,Out>::get_cuda_stream()
{
  assert( sizeof(cuda_stream) == sizeof(cudaStream_t) );
  void* ptr = &cuda_stream;
  auto stream = reinterpret_cast<cudaStream_t*>(ptr);
  return *stream;
}
#endif

} // namespace dsp::test


#endif // __dsp_TransformationTestHelper_h
