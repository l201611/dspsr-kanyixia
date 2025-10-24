/***************************************************************************
 *                                                                         *
 *   Copyright (C) 2024 by Jesmigel Cantos                                 *
 *   Licensed under the Academic Free License version 2.1                  *
 *                                                                         *
 ***************************************************************************/

#include "dsp/OutputDADABuffer.h"
#include "dsp/BitSeries.h"

#include <gtest/gtest.h>

#ifndef __dsp_OutputDADABufferTest_h
#define __dsp_OutputDADABufferTest_h

namespace dsp::test {

/**
 * @class OutputDADABufferTest
 * @brief A class for testing the OutputDADABuffer class.
 */
class OutputDADABufferTest : public ::testing::TestWithParam<const char *>
{
  public:

    /**
     * @brief Construct a new OutputDADABufferTest object
     */
    OutputDADABufferTest();

    /**
     * @brief Destroy the OutputDADABufferTest object
     */
    ~OutputDADABufferTest() = default;

    //! Number of buffers in the header
    uint64_t hdr_nbufs{1};
    //! Size of each buffer in the header
    uint64_t hdr_bufsz{4096};
    //! Number of buffers in the data
    uint64_t data_nbufs{1};
    //! Size of each buffer in the data
    uint64_t data_bufsz{4096};
    //! Number of readers
    unsigned num_readers{1};
    //! Test value verifies that DADA header is propagated
    float test_float_value{1.234};

    //! Template header used to initialize DADA header
    dsp::DADAHeader template_header;

    //! Pointer to PSRDADA header block
    ipcbuf_t header_block;
    //! Pointer to PSRDADA data block
    ipcio_t data_block;
    //! Input BitSeries container
    Reference::To<dsp::BitSeries> input{nullptr};

    //! Number of frequency channels
    unsigned nchan = 32;
    //! Number of polarizations
    unsigned npol  = 2;
    //! Number of dimensions
    unsigned ndim  = 2;
    //! Number of bits
    unsigned nbit  = 8;
    //! Number of data points
    uint64_t ndat = 32;

    // Helper variables used interacting with smrb
    //! Key for the shared memory ring buffer
    std::string smrb_key = "a000";
    //! Flag indicating whether the HDU is connected
    bool hdu_connected = false;
    //! Flag indicating whether the HDU is locked
    bool hdu_locked = false;
    //! Flag indicating whether the HDU is opened
    bool hdu_opened = false;
    //! Shared memory interface
    dada_hdu_t* hdu;

    // Helper functions
    /**
     * @brief Convert a string key to a key_t.
     * @param key_string The string key to convert.
     * @return The converted key.
     */
    key_t convert_key(const std::string &key_string);

    /**
     * @brief Setup the shared memory ring buffer.
     * @param header_key The key for the header block.
     */
    void setup_smrb(const std::string &header_key);

    /**
     * @brief Teardown the shared memory ring buffer.
     */
    void teardown_smrb();

    /**
     * @brief Initialize the input BitSeries container.
     */
    void init_input();

    /**
     * @brief Fill the input BitSeries container with data.
     */
    void fill_input();

    /**
     * @brief Clear the input BitSeries container.
     */
    void clear_input();

    /**
     * @brief Connect to the shared memory ring buffer.
     * @param smrb_key The key for the shared memory ring buffer.
     */
    void connect(std::string smrb_key);

    /**
     * @brief Disconnect from the shared memory ring buffer.
     */
    void disconnect();

    /**
     * @brief Assert the header information.
     */
    void assert_header();

    /**
     * @brief Assert the data integrity.
     */
    void assert_data();

    //! Pointer to the OutputDADABuffer object being tested
    std::shared_ptr<OutputDADABuffer> odb{nullptr};

  protected:

    void SetUp() override;

    void TearDown() override;

};

} // namespace dsp::test

#endif // __dsp_OutputDADABufferTest_h
