//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __TransferCUDA_h
#define __TransferCUDA_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <cuda_runtime.h>

namespace dsp {

  class TransferCUDA : public Transformation<TimeSeries,TimeSeries>
  {
  public:

    //! Default constructor - always out of place
    TransferCUDA(cudaStream_t _stream);

    void set_kind (cudaMemcpyKind k) { kind = k; }
    void prepare ();

    Operation::Function get_function () const { return Operation::Structural; }

    //! Worker function that performs the copy and any required synchronization
    /*! This method is static so that the logic can be reused in other places. */
    static void copy (void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);

  protected:

    //! Do stuff
    void transformation();

    cudaMemcpyKind kind;

    cudaStream_t stream;

  };

}

#endif
