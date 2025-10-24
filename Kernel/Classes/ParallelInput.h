//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/ParallelInput.h

#ifndef __dsp_Kernel_Classes_ParallelInput_h
#define __dsp_Kernel_Classes_ParallelInput_h

#include "dsp/Operation.h"
#include "Registry.h"
#include "MJD.h"

namespace dsp {

  class ParallelBitSeries;
  class Observation;
  class Input;

  //! Loads multiple parallel BitSeries data from multiple files
  class ParallelInput : public Operation
  {

  public:
  
    typedef ParallelBitSeries OutputType;

    //! Constructor
    ParallelInput (const char* name);
    
    //! Destructor
    virtual ~ParallelInput ();

    //! Return a pointer to a new instance of the appropriate sub-class
    /*! This is the entry point for creating new instances of ParallelInput objects. */
    static ParallelInput* create (const std::string& descriptor);

    //! Return the Input corresponding to the specified index
    const Input* at (unsigned index) const;

    //! Return the Input corresponding to the specified index
    Input* at (unsigned index);

    //! Return true if descriptor describes data in the recognized format
    /*! Derived classes must define the conditions under which they can be used to parse the given descriptor */
    virtual bool matches (const std::string& descriptor) const = 0;

    //! Open the inputs
    virtual void open (const std::string& descriptor) = 0;

    //! Load the current ParallelBitSeries
    void operation () override;
    
    //! Get the information about the data source
    operator const Observation* () const { return get_info(); }

    //! Get the information about the data source
    virtual Observation* get_info ();

    //! Get the information about the data source
    virtual const Observation* get_info () const;

    //! Set the ParallelBitSeries to which data will be loaded
    virtual void set_output (ParallelBitSeries*);

    //! Get the ParallelBitSeries to which data will be loaded
    ParallelBitSeries* get_output ();

    //! Prepare the output with the attributes of the data source
    void prepare () override;

    //! Reserve the maximum amount of output space required
    void reserve () override;

    //! Combine accumulated results with another operation
    void combine (const Operation*) override;

    //! Report operation statistics
    void report () const override;

    //! Reset accumulated results to intial values
    void reset () override;

    //! Reset accumulated wall time to zero
    void reset_time () const override;

    //! Reserve the maximum amount of space required in the given container
    virtual void reserve (ParallelBitSeries*);

    //! End of data
    virtual bool eod() const;

    //! Close / shutdown / free any resources, if applicable
    virtual void close ();
    
    //! Return to the start of data, if possible
    void restart ();

    //! Load BitSeries data
    /*! Only this load method is guaranteed to be thread safe */
    virtual void load (ParallelBitSeries*);

    //! Seek to the specified time sample
    virtual void seek (int64_t offset, int whence = 0);

    //! Return the first time sample to be read on the next call to operate
    uint64_t tell () const;

    //! Seek to a sample close to the specified MJD
    virtual void seek (const MJD& mjd);

    //! Convenience method used to report the offset in seconds
    double tell_seconds () const;

    //! Set the start of observation offset in units of seconds
    void set_start_seconds (double seconds);

    //! Convenience method used to set the number of seconds
    void set_total_seconds (double seconds);

    //! Return the number of time samples to load on each load_block
    virtual uint64_t get_block_size () const = 0;
    //! Set the number of time samples to load on each load_block
    virtual void set_block_size (uint64_t _size) = 0;

    //! Return the number of time samples by which consecutive blocks overlap
    virtual uint64_t get_overlap () const = 0;
    //! Set the number of time samples by which consecutive blocks overlap
    virtual void set_overlap (uint64_t _overlap) = 0;

    //! Get the time sample resolution of the data source
    virtual unsigned get_resolution () const = 0;

    //! The number of bytes of additional storage used by the operation
    uint64_t bytes_storage() const override;

    //! The number of bytes of scratch space used by the operation
    uint64_t bytes_scratch () const override;

    //! In multi-threaded programs, a mutual exclusion and a condition
    virtual void set_context (ThreadContext* _context) { context = _context; }
    bool has_context () const { return context != nullptr; }
    ThreadContext* get_context () { return context; }

    //! typedef used to simplify template syntax in ParallelInput_registry.C
    typedef Registry::List<ParallelInput> Register;

    //! Return the list of registered sub-classes
    static Register& get_register();

  protected:

    //! Information about the data source (passed on to ParallelBitSeries in load)    
    Reference::To<Observation> info;

    //! Array of parallel Input objects
    std::vector< Reference::To<Input> > inputs;

    //! The output array of parallel BitSeries
    Reference::To<ParallelBitSeries> output;
    
    //! Thread coordination used in Input::load method
    ThreadContext* context {NULL};
  };

}

#endif // !defined(__dsp_Kernel_Classes_ParallelInput_h)
  
