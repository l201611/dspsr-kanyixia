/***************************************************************************
 *
 *   Copyright (C) 2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/WeightedTimeSeries.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>

#ifndef __dsp_WeightedTimeSeriesTest_h
#define __dsp_WeightedTimeSeriesTest_h

namespace dsp::test
{
  class TestInfo
  {
  public:
    uint64_t ndat = 1024;
    unsigned nchan = 1;
    unsigned npol = 1;
    unsigned ndim = 1;

    unsigned ndat_per_weight = 16;
    unsigned nchan_weight = 1;
    unsigned npol_weight = 1;

    dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;

    bool on_gpu = false;
  };

  class WeightedTimeSeriesTest : public ::testing::TestWithParam<TestInfo>
  {
  public:

    /**
     * @brief Construct a new WeightedTimeSeriesTest object
     *
     */
    WeightedTimeSeriesTest();

    /**
     * @brief Destroy the WeightedTimeSeriesTest object
     *
     */
    ~WeightedTimeSeriesTest() = default;

  protected:

    //! Construct and initialize the device under test and any other test support infrastructure
    void SetUp() override;

    //! Destroy the device under test and any other test support infrastructure
    void TearDown() override;

    //! Current system under test
    Reference::To<dsp::WeightedTimeSeries> current_sut;

    //! System under test on host
    Reference::To<dsp::WeightedTimeSeries> sut_host;

    //! System under test on device
    Reference::To<dsp::WeightedTimeSeries> sut_device;

    //! Device memory manager
    Reference::To<dsp::Memory> device_memory;

    //! Information about the current test
    TestInfo info;

    //! Helper function for generating weights in WeightedTimeSeries
    void generate_weights();

    //! Helper function for generating floating-point data and weights in WeightedTimeSeries
    void generate_data();

    //! Helper function for testing weights in WeightedTimeSeries
    /*!
    @param expect the expected dimensions of the system under test
    @param sut option system under test overrides current_sut
    */
    void assert_weights(const TestInfo& expect, const dsp::WeightedTimeSeries* sut = 0);

    //! Helper function for testing floating-point data and weights in WeightedTimeSeries
    void assert_data(const TestInfo& expect);

    //! Helper function used when generating weights
    uint16_t expected_weight(unsigned ichan, unsigned ipol, uint64_t iweight);

    /**
     * @brief Encode the current time sample, channel, polarisation and dimension to a known floating point value.
     *
     * This method allows testing of digitisation to know we pack the value in the correct order given the
     * given time sample, channel, polarisation and dimension.
     *
     * @param idat the current time sample
     * @param ichan the current channel
     * @param ipol the current polarisation
     * @param idim the current dimension
     */
    float expected_data(uint64_t idat, unsigned ichan, unsigned ipol, unsigned idim);

#ifdef HAVE_CUDA
    //! @brief CUDA stream handle.
    cudaStream_t stream{nullptr};
#endif    
  };

} // namespace dsp::test

#endif // __dsp_WeightedTimeSeriesTest_h
