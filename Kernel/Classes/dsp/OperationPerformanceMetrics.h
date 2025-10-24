//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/OperationPerformanceMetrics.h

#ifndef __dsp_OperationPerformanceMetrics_h
#define __dsp_OperationPerformanceMetrics_h

#include "dsp/Operation.h"
#include "dsp/Observation.h"

namespace dsp {

  /**
   * @brief a class that captures the performance metrics of operations.
   *
   * Subclasses of @see Operation are expected to return the metrics of
   * per loop of a pipeline.
   */
  class Operation::PerformanceMetrics
  {
  public:    
    //! Total time spanned by the data that operation has processed in seconds
    double total_data_time = 0.0;

    //! The total amount of processing time for the operation in seconds
    double total_processing_time = 0.0;

    //! The total number of input bytes the operation has processed
    uint64_t total_bytes_processed = 0;

    void update_metrics(const Observation* input)
    {
      // This method is expected to be called by the operation
      // to accumulate the total bytes processed and data time
      // for the input samples.
      total_bytes_processed += input->get_nbytes();
      const double rate = input->get_rate();
      if (std::isfinite(rate) && rate > 0.0)
      {
        const double data_time = input->get_ndat() / rate;
        total_data_time += data_time;
      }
    }

  };

} // namespace dsp

#endif  // !defined(__dsp_OperationPerformanceMetrics_h)
