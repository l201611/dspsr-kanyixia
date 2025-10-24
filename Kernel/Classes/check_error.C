#include "Error.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

void check_error (const char* method)
{
  cudaDeviceSynchronize ();

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    cerr << method << " cudaGetLastError="
         << cudaGetErrorString (error) << endl;

    throw Error (InvalidState, method, cudaGetErrorString (error));
  }
}

void check_error_stream (const char* method, cudaStream_t stream)
{
  cudaStreamSynchronize (stream);

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    cerr << method << " cudaGetLastError="
          << cudaGetErrorString (error) << endl;

    throw Error (InvalidState, method, cudaGetErrorString (error));
  }
}
