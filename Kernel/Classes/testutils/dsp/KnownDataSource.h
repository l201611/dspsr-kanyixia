//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2025 by Will Gauvin
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#ifndef __dsp_KnownDataSource_h
#define __dsp_KnownDataSource_h

#include <dsp/TestSource.h>

namespace dsp::test {

/**
 * @brief a test data source that replays known data, and optionally weights.
 *
 * Data, and optionally weights, that is generated externally via a test's own generate method
 * can be used to a TestSource.  The idea of this is that multiple chucks of data
 * can be generated and instances of this class will ensure that the whole data is
 * split up into chunks of ndat as defined on the output timeseries that the data is
 * written to.
 */
class KnownDataSource : public TestSource {
  public:
    /**
     * @brief Construct a new KnownDataSource object
     *
     * @param niterations number of operation iterations to perform.
     */
    KnownDataSource(unsigned niterations = 1) : TestSource("KnownDataSource", niterations) {}

    /**
     * @brief Destroy the KnownDataSource object
     *
     */
    ~KnownDataSource() = default;

    //! increment the number of iterations performed.
    void operation();

    /**
     * @brief set the data to use in the output TimeSeries
     *
     * @param _data the data to use in the output TimeSeries
     */
    void set_data(std::vector<float> _data) { data = _data; }

    /**
     * @brief get the data to use in the output TimeSeries
     */
    const std::vector<float>& get_data() const { return data; }

    /**
     * @brief set the weights to use in the output WeightedTimeSeries
     *
     * @param _weights the weights to use in the output WeightedTimeSeries
     */
    void set_weights(std::vector<uint16_t> _weights) { weights = _weights; }

    /**
     * @brief the weights to use in the output WeightedTimeSeries
     */
    const std::vector<uint16_t>& get_weights() const { return weights; }

    /**
     * @brief Return a default constructed clone of self.
     *
     * @return Source* clone of this object.
     */
    Source* clone() const;

  private:

    //! The data to use when updating the output TimeSeries
    std::vector<float> data{};

    //! The weights to use when updating the output WeightedTimeSeries
    std::vector<uint16_t> weights{};

};

} // namespace dsp::test

#endif // !defined(__dsp_KnownDataSource_h)
