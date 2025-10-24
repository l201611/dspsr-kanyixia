//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Generators/dsp/Generator.h

#ifndef __dsp_Signal_Generators_Generator_h
#define __dsp_Signal_Generators_Generator_h

#include "dsp/Source.h"

namespace dsp {

  //! Base class of artificial signal generators
  class Generator : public Source
  {
  protected:

    //! The information about the source
    Reference::To<Observation> info;

    //! The output TimeSeries
    Reference::To<TimeSeries> output;

    //! The output order
    TimeSeries::Order order = TimeSeries::OrderFPT;

    //! The current time sample
    uint64_t current_sample = 0;

    //! The number of time samples output on each operation
    uint64_t block_size = 0;

    //! The number of time samples by which output blocks overlap
    uint64_t overlap = 0;
   
  public:

    //! Constructor
    Generator (const char* name) : Source(name) { info = new Observation; }

    //! Destructor
    ~Generator() = default;

    //! Get the const Observation attributes that describe the source
    const Observation* get_info() const override { return info; }

    //! Get the mutable Observation attributes that describe the source
    Observation* get_info() override { return info; }

    //! Set the TimeSeries object used to store output data
    void set_output (TimeSeries* data) override { output = data; }

    //! Get the TimeSeries object used to store output data
    TimeSeries* get_output () override { return output; }

    //! Return true if this object has a TimeSeries object to store output data
    bool has_output () const override { return output; }

    //! Get the total number of time samples available (return 0 if unknown)
    uint64_t get_total_samples () const override { return info->get_ndat(); }

    //! Get the current time sample offset from the start of data
    uint64_t get_current_sample () const override { return current_sample; }

    //! Get the current time offset from the start of data in seconds
    double get_current_time () const override 
    { return current_sample / info->get_rate(); }

    //! Seek to the specified time in seconds
    /*! An exception is thrown if unable to seek. */
    void seek_time (double second) override
    { current_sample = round(second*info->get_rate()); }

    //! Truncate the time series at the specified second
    /*! An exception is thrown if unable to set total samples. */
    void set_total_time (double second) override
    { info->set_ndat(round(second*info->get_rate())); }

    //! Return true when the end of data has been reached
    bool end_of_data () const override
    { return current_sample >= info->get_ndat(); }

    //! By default, signal generators do nothing when closed
    void close () override { /* do nothing */ }

    //! Return to the start of data, if possible
    void restart () override { current_sample = 0; }

    //! Set the desired number of time samples per segment output by the source
    void set_block_size (uint64_t n) override { block_size = n; }

    //! Get the number of time samples per segment output by the source
    /*! This may not be the same as the number specified by set_block_size owing to the intrinsic resolution of the source */
    uint64_t get_block_size () const override { return block_size; }

    //! Set the number of time samples by which segments of output TimeSeries data overlap
    void set_overlap (uint64_t n) override { overlap = n; }

    //! Get the number of time samples by which segments of output TimeSeries data overlap
    uint64_t get_overlap () const override { return overlap; }

    //! By default, signal generators support either order
    bool get_order_supported (TimeSeries::Order) const { return true; }

    //! Set the order of the dimensions in the output TimeSeries
    void set_output_order (TimeSeries::Order _order) override { order = _order; }

    //! By default, signal generators operate only only host memory
    bool get_device_supported (Memory*) const override;

    //! By default, signal generators operate only only host memory
    void set_device (Memory*) override { /* do nothing */}

    //! By default, signal generators share no resources
    void share (Source*) override { /* do nothing */}

    //! By default, there are no shared resources to protect
    void set_context (ThreadContext* context) override { /* do nothing */}
  };

}

#endif // !defined(__dsp_Signal_Generators_Generator_h)
