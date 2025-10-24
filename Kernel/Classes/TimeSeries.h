//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/TimeSeries.h

#ifndef __TimeSeries_h
#define __TimeSeries_h

#include <memory>

#include "Error.h"

#include "dsp/DataSeries.h"

namespace dsp {

  //! Arrays of consecutive samples for each polarization and frequency channel
  class TimeSeries : public DataSeries
  {
  public:

    //! Order of the dimensions
    enum Order {

      //! Frequency, Polarization, Time (default before 3 October 2008)
      OrderFPT,

      //! Time, Frequency, Polarization (better for many things)
      OrderTFP

    };

    //! Automatically delete arrays on resize(0)
    static bool auto_delete;

    //! Null constructor
    TimeSeries ();

    //! Copy constructor
    TimeSeries(const TimeSeries& ts);

    //! Destructor
    virtual ~TimeSeries();

    //! Assignment operator
    const TimeSeries& operator = (const TimeSeries& copy);

    //! Clone operator
    TimeSeries* clone() const override;

    //! Returns a null-instantiation (calls new)
    TimeSeries* null_clone() const override;

    //! Swaps the two TimeSeries if DataSeries is a TimeSeries.  Returns '*this'
    TimeSeries& swap_data (DataSeries& ts) override;

    //! Swaps the two TimeSeries's.  Returns '*this'
    virtual TimeSeries& swap_data (TimeSeries& ts);

    //! Call TimeSeries::copy if Observation is a TimeSeries
    void copy (const Observation*) override;

    //! Call TimeSeries::copy if DataSeries is a TimeSeries
    void copy (const DataSeries*) override;

    //! Copy all TimeSeries attributes
    virtual void copy (const TimeSeries*);

    //! Add each value in data to this
    virtual void add (const TimeSeries*);

    //! Multiply each value by this scalar
    virtual void multiply (float mult);

    //! Get the order
    Order get_order () const;

    //! Set the order
    void set_order (Order order);

    //! Copy the dimensions of another TimeSeries instance
    void copy_dimensions (const Observation* copy) override;

    //! Copy the transient attributes of another observation
    void copy_transient_attributes (const Observation*) override;

    //! Copy the data of another TimeSeries instance
    virtual void copy_data (const TimeSeries* data, uint64_t idat_start = 0, uint64_t ndat = 0);

    //! Match the internal memory layout of another TimeSeries
    virtual void internal_match (const TimeSeries*);

    //! Match the internal memory layout of another TimeSeries
    void internal_match (const DataSeries*) override;

    //! Disable the set_nbit method of the Observation base class
    void set_nbit (unsigned) override;

    //! Allocate the space required to store nsamples time samples.
    void resize (uint64_t nsamples) override;

    //! Decrease the array lengths without changing the base pointers
    virtual void decrease_ndat (uint64_t new_ndat);

    //! For nchan=1, npol=1 data this uses the data in 'buffer'
    TimeSeries& use_data(float* _buffer, uint64_t _ndat);

    //! Return pointer to the specified data block
    float* get_datptr (unsigned ichan=0, unsigned ipol=0);

    //! Return pointer to the specified data block
    const float* get_datptr (unsigned ichan=0, unsigned ipol=0) const;

    //! Return pointer to the specified data block
    float* get_dattfp ();

    //! Return pointer to the specified data block
    const float* get_dattfp () const;

    //! Offset the base pointer by offset time samples
    virtual void seek (int64_t offset);

    //! Append the given TimeSeries to the end of 'this'
    //! Copy data from given TimeSeries in front of the current position
    void prepend (const dsp::TimeSeries*, uint64_t pre_ndat = 0);

    //! Return the sample offset from the start of the data source
    int64_t get_input_sample () const { return input_sample; }

    //! Used to arrange pieces in order during input buffering
    void set_input_sample (uint64_t sample) { input_sample = sample; }

    //! Get the span (number of floats)
    uint64_t get_nfloat_span () const;

    //! Return the mean of the data for the specified channel and poln
    double mean (unsigned ichan, unsigned ipol);

    //! Check that each floating point value is roughly as expected
    virtual void check (float min=-10.0, float max=10.0);

    //! Set the zeroed data flag
    void set_zeroed_data (bool _zeroed_data) { zeroed_data = _zeroed_data; }

    //! Get the zeroed data flag
    bool get_zeroed_data () const { return zeroed_data; }

    void finite_check () const;

    //! Link that->reserve_ndat to this->reserve_ndat
    /*! that->reserve_ndat is updated whenever this->change_reserve is called */
    void set_match (TimeSeries* that);

    //! Abstract base class of alternative implementations of certain TimeSeries methods
    class Engine;

    //! Set an alternative implementation of certain TimeSeries methods
    void set_engine (Engine*);

    //! Set the alternative implementation of certain TimeSeries methods
    Engine* get_engine () const { return engine; };

    //! For FPT-ordered data, returns the number of floats between consecutive blocks of time samples
    uint64_t get_stride () const;

  protected:

    //! Returns a uchar pointer to the first piece of data
    unsigned char* get_data() override;
    //! Returns a uchar pointer to the first piece of data
    const unsigned char* get_data() const override;

    virtual void prepend_checks (const TimeSeries*, uint64_t pre_ndat);

    //! Pointer into buffer, offset to the first time sample requested by user
    float* data;

    //! Change the amount of memory reserved at the start of the buffer
    void change_reserve (int64_t change) const;

    //! Get the amount of memory reserved at the start of the buffer
    uint64_t get_reserve () const { return reserve_ndat; }

    //! Flag for whether the data contains zero values. See ZapWeight
    bool zeroed_data;

    friend class Reserve;
    friend class Unpacker;

    // do the work of the null_clone: copy necessary attributes from the given TimeSeries
    void null_work (const TimeSeries* from);

    Reference::To<Engine> engine;

  private:

    //! Order of the dimensions
    Order order;

    //! Reserve space for this many timesamples preceding the base address
    uint64_t reserve_ndat;

    //! Number of floats reserved
    uint64_t reserve_nfloat;

    //! TimeSeries that should match this one internally
    Reference::To<TimeSeries, false> match;

    //! Sample offset from start of source
    /*! Set by Unpacker class and used by multithreaded InputBuffering */
    int64_t input_sample;

    //! Called by constructor to initialise variables
    void init ();


  };

  class TimeSeries::Engine : public OwnStream
  {
  public:

    virtual void prepare (dsp::TimeSeries * to) = 0;

    virtual void prepare_buffer (unsigned nbytes) = 0;

    virtual void copy_data_fpt (const dsp::TimeSeries * copy,
                                uint64_t idat_start = 0,
                                uint64_t ndat = 0) = 0;

  };

}

#endif
