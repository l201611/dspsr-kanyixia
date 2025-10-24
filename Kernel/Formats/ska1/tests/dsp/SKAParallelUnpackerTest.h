/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ASCIIObservation.h"
#include "dsp/SKAParallelUnpacker.h"
#include "dsp/BitSeries.h"
#include "dsp/WeightedTimeSeries.h"

#include <gtest/gtest.h>
#include <string>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifndef __dsp_ska1_SKAParallelUnpackerTest_h
#define __dsp_ska1_SKAParallelUnpackerTest_h

namespace dsp::test {

 /**
   * @brief a struct to keep track of all the parameters used with the tests.
   *
   * This is generated during setup of parameterised tests to allow different
   * combinations of:
   *  use the CPU or GPU unpacker
   *  CBF version/format to assume (low, mid)
   */
  struct TestParam {

    //! Whether the test should run on the GPU or not
    bool on_gpu;

    //! Name of the CBF data format
    std::string cbf_name;

  } test_param_t;

class SKAParallelUnpackerTest : public ::testing::TestWithParam<TestParam>
{
  public:
    /**
     * @brief Construct a new SKAParallelUnpackerTest object
     */
    SKAParallelUnpackerTest();

    /**
     * @brief Destroy the SKAParallelUnpackerTest object
     */
    ~SKAParallelUnpackerTest() = default;

    /**
     * @brief Construct and configure the dsp::SKAParallelUnpacker object to be tested
     *
    */
    dsp::SKAParallelUnpacker* new_device_under_test();

    /**
     * @brief Generate test data for input to SKAParallelUnpacker
     *
     */
    void generate_data();

    /**
     * @brief Helper function for asserting output dsp::TimeSeries data
     *
     */
    void assert_output();

    /**
     * @brief Helper function setting the input and output containers used by the SKAParallelUnpacker
     *
     * @param spu SKAParallelUnpacker shared pointer
     *
     */
    void set_input_output(std::shared_ptr<dsp::SKAParallelUnpacker> spu);

    /**
     * @brief Helper function for calling the configure method of the SKAParallelUnpacker
     *
     * @param spu SKAParallelUnpacker shared pointer
     *
     */
     void call_configure(std::shared_ptr<dsp::SKAParallelUnpacker> spu);

    /**
     * @brief Helper function for calling the prepare method of the SKAParallelUnpacker
     *
     * @param spu SKAParallelUnpacker shared pointer
     *
     */
     void call_prepare(std::shared_ptr<dsp::SKAParallelUnpacker> spu);

    /**
     * @brief Helper function for calling the reserve method of the SKAParallelUnpacker
     *
     * @param spu SKAParallelUnpacker shared pointer
     *
     */
     void call_reserve(std::shared_ptr<dsp::SKAParallelUnpacker> spu);

    /**
     * @brief Helper function for calling the operate method of the SKAParallelUnpacker
     *
     * @param spu SKAParallelUnpacker shared pointer
     *
     */
     void call_operate(std::shared_ptr<dsp::SKAParallelUnpacker> spu);

    /**
     * @brief Helper function for calling the reset method of the SKAParallelUnpacker
     *
     * @param spu SKAParallelUnpacker shared pointer
     *
     */
     void call_reset(std::shared_ptr<dsp::SKAParallelUnpacker> spu);

    /**
     * @brief Helper function for loading the contents of a file into a string variable.
     *
     * @param file_name file name to be loaded
     *
     * @return std::string content of file selected
     *
     */
    std::string load_config_from_file(const std::string& file_name);

    //! input container
    Reference::To<dsp::ParallelBitSeries> input;

    //! output container
    Reference::To<dsp::TimeSeries> output;

    //! input to device container
    Reference::To<dsp::ParallelBitSeries> device_input;

    //! output of device container
    Reference::To<dsp::WeightedTimeSeries> device_output;

    //! device memory manager
    Reference::To<dsp::Memory> device_memory;

    //! observation configuration for data
    Reference::To<dsp::ASCIIObservation> obs_data;

    //! observation configuration for weights
    Reference::To<dsp::ASCIIObservation> obs_weights;

    //! number of bit-series in the parallel input
    const unsigned nbitseries{2};

    //! number of channels
    unsigned nchan{0};

    //! number of polarisations
    unsigned npol{0};

    //! number of dimensions
    unsigned ndim{0};

    //! number of time samples
    uint64_t ndat{96};

    //! number of bits
    uint64_t nbit{0};

    //! number of time samples per packet
    uint32_t nsamp_per_packet{0};

    //! number of frequency channels per packet
    uint32_t nchan_per_packet{0};

  protected:

    /**
     * @brief Sets up the test environment.
     *
     */
    void SetUp() override;

    /**
     * @brief Tears down the test environment.
     *
     */
    void TearDown() override;

    /**
     * @brief Get the expected scale value for the specified packet within a heap.
     *
     * @param ipacket packet within the heap of packets that span all channels
     * @return uint32_t expected scale value
     */
    uint32_t get_expected_scale(uint32_t ipacket);

    /**
     * @brief Get the expected 16-bit data value for the sample
     *
     * @param ival the value (idat * ndim + idat) within a packet
     * @param ipol the polarisation of the sample
     * @param scale the scale factor apply
     * @return int16_t the expected data value
     */
    int16_t get_expected_data_16b(uint32_t ival, uint32_t ipol, uint32_t scale);

    /**
     * @brief Get the expected 8-bit data value for the sample
     *
     * @param ival the value (idat * ndim + idat) within a packet
     * @param ipol the polarisation of the sample
     * @param scale the scale factor apply
     * @return int8_t the expected data value
     */
    int8_t get_expected_data_8b(uint32_t ival, uint32_t ipol, uint32_t scale);

    /**
     * @brief Get the expected weight value for a channel
     *
     * @param ichan the channel to return the weight for
     * @return uint16_t the expected weight value
     */
    uint16_t get_expected_weight(uint32_t ichan);

    //! Set true when test should be performed on GPU
    bool on_gpu{false};

    //! Name of the CBF packet format under test
    std::string cbf_name;

    //! Order of TimeSeries data
    dsp::TimeSeries::Order order = dsp::TimeSeries::OrderFPT;

#ifdef HAVE_CUDA
    //! @brief CUDA stream handle.
    cudaStream_t stream{nullptr};
#endif
};

} // namespace dsp::test

#endif // __dsp_ska1_SKAParallelUnpackerTest_h
