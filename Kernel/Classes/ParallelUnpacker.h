//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/ParallelUnpacker.h

#ifndef __ParallelUnpacker_h
#define __ParallelUnpacker_h

#include "dsp/ParallelBitSeries.h"
#include "dsp/Unpacker.h"
#include "Registry.h"

namespace dsp {

  class ParallelInput;

  //! Manages an array of Unpacker objects to be process in parallel
  class ParallelUnpacker : public Transformation <ParallelBitSeries, TimeSeries>
  {

  public:

    //! Constructor
    ParallelUnpacker (const char* name);

    //! Return a pointer to a new instance of the appropriate sub-class
    static ParallelUnpacker* create (const Observation* observation);

    //! Clone operator
    ParallelUnpacker* clone() const;

    //! Return true if the unpackers support the specified output order
    bool get_order_supported (TimeSeries::Order) const;

    //! Set the order of the dimensions in the output TimeSeries
    void set_output_order (TimeSeries::Order);

    //! Return true if the unpackers can operate on the specified device
    virtual bool get_device_supported (Memory*) const;

    //! Set the device on which the unpacker will operate
    virtual void set_device (Memory*);

    //! Return true if the derived class can convert the Observation
    /*! Derived classes must define the conditions under which they can be used to parse the given data. */
    virtual bool matches (const Observation* observation) const = 0;

    //! Specialize the unpackers for the Observation
    virtual void match (const Observation* observation) = 0;

    //! Match the unpacker to the resolution of the Input
    /*! The the first Input of ParallelInput is passed to the match_resolution method
        of the first Unpacker in this ParallelUnpacker. */
    virtual void match_resolution (ParallelInput*);

    //! Return the smallest number of time samples that can be unpacked
    /*! The resolution of the first Unpacker is returned. */
    virtual unsigned get_resolution () const;

    //! Copy the input attributes to the output
    void prepare () override;

    //! Reserve the maximum amount of space required in the output
    void reserve () override;

    //! The number of bytes of additional storage used by the operation
    uint64_t bytes_storage() const override;

    //! The number of bytes of scratch space used by the operation
    uint64_t bytes_scratch () const override;

    //! Set the policy for buffering input and/or output data
    void set_buffering_policy (BufferingPolicy* policy) override;

    //! Set verbosity ostream
    void set_cerr (std::ostream& os) const override;

    //! typedef used to simplify template syntax in ParallelUnpacker_registry.C
    typedef Registry::List<ParallelUnpacker> Register;

    //! Return the list of registered sub-classes
    static Register& get_register();

   protected:

    //! The order of the dimensions in the output TimeSeries
    TimeSeries::Order output_order;

    //! The unpacking routine
    /*! This method must unpack the data from the ParallelBitSeries Input into the TimeSeries output. */
    virtual void unpack () = 0;

    //! The operation unpacks parallel BitSeries into floating point TimeSeries
    void transformation () override;

    //! The parallel Unpackers
    std::vector< Reference::To<Unpacker> > unpackers;

  };

}

#endif // !defined(__ParallelUnpacker_h)

