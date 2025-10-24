//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __dsp_SumSource_h
#define __dsp_SumSource_h

#include <dsp/Source.h>
#include <dsp/TimeSeries.h>

namespace dsp::test {

/**
 * @brief Class that can be used to combine multiple input sources.
 *
 * Instances of this source should be configured with multiple sources that can be added together
 * to make a more complex source.  An example of using this is to add a @see GaussianNoiseSource along
 * with a @see ImpulsiveNoiseSource, because the ImpulsiveNoiseSource does not provide background but
 * is a pure signal.
 */
class SumSource : public dsp::Source
{
  public:
    /**
     * @brief Construct a new SumSource object
     *
     * @param niterations number of operation iterations to perform.
     */
    SumSource(unsigned niterations = 1);

    /**
     * @brief Destroy the SumSource object
     *
     */
    virtual ~SumSource() = default;

    //! increment the number of iterations performed.
    void operation() override;

    /**
     * @brief Return a default constructed clone of self.
     *
     * @return Source* clone of this object.
     */
    dsp::Source* clone() const override;

    //! Get the const Observation attributes that describe the source
    dsp::Observation* get_info() const override { return info; };

    //! Get the mutable Observation attributes that describe the source
    dsp::Observation* get_info() override { return info; };

    /**
     * @brief Set the TimeSeries object that is used to provide output to callers of the get_output method.
     *
     * Note: this timeseries must be configured with meta-data, resized and filled with data by the caller to this method.
     *
     * @param _output TimeSeries that will be returned by get_output
     */
    void set_output(dsp::TimeSeries * _output) override;

    //! Get the TimeSeries used to store output data
    dsp::TimeSeries * get_output() override { return output; };

    //! Return true if this object has a TimeSeries object to store output data
    bool has_output() const override { return output; };

    //! Get the total number of time samples available (return 0 if unknown)
    uint64_t get_total_samples() const override { return total_samples; };

    //! Get the current time sample offset from the start of data
    uint64_t get_current_sample() const override { return current_samples; };

    //! Get the current time offset from the start of data in seconds
    double get_current_time() const override { return second; };

    //! Seek to the specified time in seconds
    /*! An exception is thrown if unable to seek. */
    void seek_time(double second) override;

    //! Truncate the time series at the specified second
    /*! An exception is thrown if unable to set total samples. */
    void set_total_time(double _second) override;

    //! Return true if the end of data has been reached
    bool end_of_data() const override { return eod; };

    //! Return to the start of data, if possible
    void restart() override;

    //! Close / shutdown / free any resources, if applicable
    void close() override;

    //! Set the desired number of time samples per segment output by the source
    void set_block_size(uint64_t _block_size) override;

    //! Get the number of time samples per segment output by the source
    /*! This may not be the same as the number specified by set_block_size owing to the intrinsic resolution of the source */
    uint64_t get_block_size() const override { return block_size; };

    //! Set the number of time samples by which segments of output TimeSeries data overlap
    void set_overlap(uint64_t _overlap) override;

    //! Get the number of time samples by which segments of output TimeSeries data overlap
    uint64_t get_overlap() const override { return overlap; };

    //! Return true if the source supports the specified output order
    bool get_order_supported(dsp::TimeSeries::Order _order) const override;

    //! Set the order of the dimensions in the output TimeSeries
    void set_output_order(dsp::TimeSeries::Order _order) override;

    //! Return true if the source can operate on the specified device
    bool get_device_supported(dsp::Memory* _device_memory) const override;

    //! Set the device on which the source will operate
    void set_device(dsp::Memory* _device_memory) override;

    //! Share any resources that can/should be shared between threads
    void share(dsp::Source* source) override;

    //! Set the mutual exclusion and condition used to protect shared resources
    void set_context(ThreadContext* context) override;

    /**
     * @brief add a source to that should be combined with other sources.
     *
     * @param _source the source to combine with other sources.
     */
    void add_source(dsp::Source* _source);

  private:
    //! the Observation attributes that describe the source
    dsp::Observation * info = nullptr;

    //! the output time series that data will be written to
    dsp::TimeSeries * output = nullptr;

    //! the total number of time samples available
    uint64_t total_samples = 0;

    //! the current time sample offset from the start of data
    uint64_t current_samples = 0;

    //! the current time offset from the start of data in seconds
    double second = 0;

    //! indicator whether the end of data has been reached or not
    bool eod = false;

    //! the desired number of time samples per segment output by the source
    uint64_t block_size = 0;

    //! the number of time samples by which segments of output TimeSeries data overlap
    uint64_t overlap = 0;

    //! the ordering of the output timeseries data
    dsp::TimeSeries::Order output_order = dsp::TimeSeries::Order::OrderFPT;

    //! pointer to a Memory object used for allocating memory
    dsp::Memory * memory = nullptr;

    //! count of the number of iterations executed by the operation method
    unsigned iterations = 0;

    //! number of iterations the operation method should perform
    unsigned niterations = 1;

    //! the sources
    std::vector< Reference::To<dsp::Source> > sources;

    //! update end of data, based on state of the sources
    void update_end_of_data();
};

} // namespace dsp

#endif // !defined(__dsp_SumSource_h)
