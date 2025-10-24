/***************************************************************************
 *
 *   Copyright (C) 2024 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <string>

#include <dsp/Source.h>
#include <dsp/TimeSeries.h>

#ifndef __dsp_TestSource_h
#define __dsp_TestSource_h

namespace dsp::test {

/**
 * @brief Class that imitates a Source of data, useful as a test harness for pipelines.
 *
 */
class TestSource : public Source
{
  public:
    /**
     * @brief Construct a new TestSource object
     *
     * @param niterations number of operation iterations to perform.
     */
    TestSource(unsigned niterations = 1);

    /**
     * @brief Construct a new TestSource object
     *
     * @param name the name of the source.
     * @param niterations number of operation iterations to perform.
     */
    TestSource(const char* name, unsigned niterations = 1);

    /**
     * @brief Destroy the TestSource object
     *
     */
    virtual ~TestSource() = default;

    //! increment the number of iterations performed.
    virtual void operation();

    /**
     * @brief Return a default constructed clone of self.
     *
     * @return Source* clone of this object.
     */
    virtual Source* clone() const;

    //! Get the const Observation attributes that describe the source
    Observation* get_info() const { return info; };

    //! Get the mutable Observation attributes that describe the source
    Observation* get_info() { return info; };

    /**
     * @brief Set the TimeSeries object that is used to provide output to callers of the get_output method.
     *
     * Note: this timeseries must be configured with meta-data, resized and filled with data by the caller to this method.
     *
     * @param _output TimeSeries that will be returned by get_output
     */
    virtual void set_output(TimeSeries * _output);

    //! Get the TimeSeries object used to store output data
    TimeSeries * get_output() { return output; };

    //! Return true if this object has a TimeSeries object to store output data
    bool has_output() const { return output; };

    uint64_t set_total_samples(uint64_t _total_samples) { return total_samples = _total_samples; };

    //! Get the total number of time samples available (return 0 if unknown)
    uint64_t get_total_samples() const { return total_samples; };

    uint64_t set_current_samples(uint64_t _current_samples) { return current_samples = _current_samples; };
    uint64_t get_current_sample() const { return current_samples; };

    double get_current_time() const { return second; };
    void seek_time(double second) {};
    void set_total_time(double _second) { second = _second; };

    bool end_of_data() const { return eod; };
    void set_end_of_data(bool _eod) { eod = _eod; };
    void restart() {};
    void close() {};

    void set_block_size(uint64_t _block_size) { block_size = _block_size; };
    uint64_t get_block_size() const { return block_size; };

    void set_overlap(uint64_t _overlap) { overlap = _overlap; };
    uint64_t get_overlap() const { return overlap; };

    bool get_order_supported(TimeSeries::Order) const { return output_order; };
    void set_output_order(TimeSeries::Order _order);

    virtual bool get_device_supported (Memory*) const { return false; };
    virtual void set_device (Memory* _device_memory) { memory = _device_memory; }

    void share(Source* source) {};
    void set_context(ThreadContext* context) {};

  protected:
    //! the Observation attributes that describe the source
    Observation * info = nullptr;

    //! the output time series that data will be written to
    TimeSeries * output = nullptr;

    //! count of the number of iterations executed by the operation method
    unsigned iterations = 0;

    //! number of iterations the operation method should perform
    unsigned niterations = 1;

    //! the ordering of the output timeseries data
    dsp::TimeSeries::Order output_order = dsp::TimeSeries::Order::OrderFPT;

    //! the total number of time samples available
    uint64_t total_samples = 0;

    //! the current time sample offset from the start of data
    uint64_t current_samples = 0;

    //! indicator whether the end of data has been reached or not
    bool eod = false;

  private:
    //! the current time offset from the start of data in seconds
    double second = 0;

    //! the desired number of time samples per segment output by the source
    uint64_t block_size = 0;

    //! the number of time samples by which segments of output TimeSeries data overlap
    uint64_t overlap = 0;

    //! pointer to a Memory object used for allocating memory
    dsp::Memory * memory = nullptr;
};

} // namespace dsp::test

#endif
