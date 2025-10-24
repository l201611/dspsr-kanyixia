//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2023 Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Kernel/Classes/dsp/InputSource.h

#ifndef __dsp_Kernel_Classes_InputSource_h
#define __dsp_Kernel_Classes_InputSource_h

#include "dsp/Source.h"
#include "templates.h"

namespace dsp {

  //! Source objects that have an input Type
  template<class Type>
  class InputSource : public Source
  {

  public:

    //! Constructor
    InputSource (const char* name) : Source(name) {}

    //! Set the input
    virtual void set_input (Type* input) = 0;

    //! Return pointer to the input
    virtual const Type* get_input () const = 0;

    //! Return pointer to the input
    virtual Type* get_input () = 0;

    //! Get the Observation attributes that describe the source
    const Observation* get_info() const override { return get_input()->get_info(); }

    //! Get the Observation attributes that describe the source
    Observation* get_info() override { return get_input()->get_info(); }

    //! Get the total number of time samples available (return 0 if unknown)
    uint64_t get_total_samples () const override { return get_info()->get_ndat(); }

    //! Get the current time sample offset from the start
    uint64_t get_current_sample () const override { return get_input()->tell(); }

    //! Get the current time offset from the start in seconds
    double get_current_time () const override { return get_input()->tell_seconds(); }

    //! Seek to the specified time in seconds
    void seek_time (double second) override { return get_input()->set_start_seconds(second); }

    //! Truncate the time series at the specified second
    void set_total_time (double second) override { return get_input()->set_total_seconds(second); }

    //! Return true when the end of data has been reached
    bool end_of_data () const override { return get_input()->eod(); }

    //! Close the input
    void close () override { get_input()->close(); }

    //! Start again from the beginning, if possible
    void restart () override { get_input()->restart(); }

    //! Set the desired number of time samples per segment output by the source
    void set_block_size (uint64_t block_size) override { get_input()->set_block_size(multiple_greater(block_size,get_resolution())); }

    //! Get the number of time samples per segment output by the source
    uint64_t get_block_size () const override { return get_input()->get_block_size(); }

    //! Set the number of time samples by which output segments should overlap
    void set_overlap (uint64_t overlap) override { get_input()->set_overlap(multiple_greater(overlap,get_resolution())); }

    //! Set the number of time samples by which segments of output TimeSeries data overlap
    uint64_t get_overlap () const override { return get_input()->get_overlap(); }

    //! Get the minimum number of time samples that can be output by the source
    virtual uint64_t get_resolution () const { return get_input()->get_resolution(); }

    //! Share any resources that can/should be shared between threads
    void share (Source* that) override
    {
      auto same = dynamic_cast<InputSource*>(that);
      if (!same)
        throw Error (InvalidParam, "dsp::InputSource::share",
        "mismatch between this type=%s and that type=%s", this->get_name().c_str(), that->get_name().c_str());

      set_input(same->get_input());
    }

    //! In multi-threaded programs, a mutual exclusion and a condition
    void set_context (ThreadContext* context) override { get_input()->set_context(context); }
  };

}


#endif // !defined(__dsp_Kernel_Classes_InputSource_h)
