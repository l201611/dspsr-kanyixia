//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/Source.h

#ifndef __dsp_Kernel_Classes_Source_h
#define __dsp_Kernel_Classes_Source_h

#include "dsp/Operation.h"
#include "dsp/TimeSeries.h"

class ThreadContext;

namespace dsp {

  //! Abstract interface to sources of TimeSeries data
  class Source : public Operation
  {

  public:

    //! Constructor
    Source (const char* name) : Operation(name) {}

    //! Destructor
    virtual ~Source () {}

    //! Return a default constructed clone of self
    virtual Source* clone() const = 0;

    //! Each Source object is classified as a Producer Operation
    Operation::Function get_function () const override { return Operation::Producer; }

    //! Get the const Observation attributes that describe the source
    virtual const Observation* get_info() const = 0;

    //! Get the mutable Observation attributes that describe the source
    virtual Observation* get_info() = 0;

    //! Set the TimeSeries object used to store output data
    virtual void set_output (TimeSeries* data) = 0;

    //! Get the TimeSeries object used to store output data
    virtual TimeSeries* get_output () = 0;

    //! Return true if this object has a TimeSeries object to store output data
    virtual bool has_output () const = 0;

    //! Get the total number of time samples available (return 0 if unknown)
    virtual uint64_t get_total_samples () const = 0;

    //! Get the current time sample offset from the start of data
    virtual uint64_t get_current_sample () const = 0;

    //! Get the current time offset from the start of data in seconds
    virtual double get_current_time () const = 0;

    //! Seek to the specified time in seconds
    /*! An exception is thrown if unable to seek. */
    virtual void seek_time (double second) = 0;

    //! Truncate the time series at the specified second
    /*! An exception is thrown if unable to set total samples. */
    virtual void set_total_time (double second) = 0;

    //! Return true when the end of data has been reached
    virtual bool end_of_data () const = 0;

    //! Close / shutdown / free any resources, if applicable
    virtual void close () = 0;

    //! Return to the start of data, if possible
    virtual void restart () = 0;

    //! Set the desired number of time samples per segment output by the source
    virtual void set_block_size (uint64_t) = 0;

    //! Get the number of time samples per segment output by the source
    /*! This may not be the same as the number specified by set_block_size owing to the intrinsic resolution of the source */
    virtual uint64_t get_block_size () const = 0;

    //! Set the number of time samples by which segments of output TimeSeries data overlap
    virtual void set_overlap (uint64_t) = 0;

    //! Get the number of time samples by which segments of output TimeSeries data overlap
    virtual uint64_t get_overlap () const = 0;

    //! Return true if the source supports the specified output order
    virtual bool get_order_supported (TimeSeries::Order) const = 0;

    //! Set the order of the dimensions in the output TimeSeries
    virtual void set_output_order (TimeSeries::Order) = 0;

    //! Return true if the source can operate on the specified device
    /*! Children of the pure-virtual Memory base class are used as a proxy for device type.
        For example:
        A) if a pointer to MemoryHost is passed and the Source supports operation
	   on CPU (host), then this function should return true;
        B) if a pointer to MemoryCUDA is passed and the Source does not support operation
	   on GPU (device), then this function should return false.
    */
    virtual bool get_device_supported (Memory*) const = 0;

    //! Set the device on which the source will operate
    /*! See get_device_supported for notes on use of Memory as a proxy for device type. */
    virtual void set_device (Memory*) = 0;

    //! Share any resources that can/should be shared between threads
    virtual void share (Source*) = 0;

    //! Set the mutual exclusion and condition used to protect shared resources
    virtual void set_context (ThreadContext* context) = 0;
  };

}

#endif // !defined(__dsp_Kernel_Classes_Source_h)
