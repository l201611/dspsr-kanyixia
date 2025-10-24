/***************************************************************************
 *
 *   Copyright (C) 2024 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADAOutputFile.h"
#include "dsp/BitSeries.h"

#include <gtest/gtest.h>
#include <random>
#include <array>

#ifndef __dsp_DADAOutputFileTest_h
#define __dsp_DADAOutputFileTest_h

namespace dsp::test
{
  class DADAOutputFileTest : public ::testing::TestWithParam<const char *>
  {
  public:
    /**
     * @brief Construct a new DADAOutputFileTest object
     *
     */
    DADAOutputFileTest();

    /**
     * @brief Destroy the DADAOutputFileTest object
     *
     */
    ~DADAOutputFileTest() = default;

    /**
     * @brief Verifies that DADA file header meta-data values match expected values.
     *
     * This method will open the file, check the meta-data and return a file
     * descriptor of the opened file, positioned at the end of the meta-data.
     *
     */
    void assert_header();

    /**
     * @brief Verifies that DADA file data match expected values from the input_data attribute.
     *
     */
    void assert_data();

    /**
     * @brief helper function for performing DADAOutputFile operation.
     * performs a function call of set_input, prepare, and operate
     *
     * @return false if an error is encountered
     *
     */
    bool perform_transform(std::shared_ptr<dsp::DADAOutputFile> odf, unsigned num_iterations);

    //! input container
    Reference::To<dsp::BitSeries> input{nullptr};

    //! number of bits per sample
    unsigned nbit{8};

    //! number of channels
    unsigned nchan{32};

    //! number of polarisations
    unsigned npol{2};

    //! number of dimensions
    unsigned ndim{2};

    //! number of time samples
    uint64_t ndat{32};

    //! sampling interval in microseconds
    double tsamp{64};

    //! centre frequency in megahertz
    double freq{1024};

    //! bandwidth in megahertz
    double bw{1};

    //! start time of the observation
    MJD start_time{12345.0f};

    //! vector for buffering generated input bit series data that span multiple transformations
    std::vector<int8_t> input_data;

  protected:

    void SetUp() override;

    void TearDown() override;

    //! Generate a uniform distribution of 8-bit signed integers of the specified length
    void generate_random_data(int8_t* data, size_t data_length);

    //! Set up random input bitseries data of the required dimensions
    void setup_random_data(unsigned num_iterations);

    //! Copy known input data from known_input_data array to the input BitSeries
    void setup_known_data(unsigned num_iterations);

    //! random device to use in seeding the value generator
    std::random_device rd{};

    //! random number generator to use in the normal distribution generation
    std::mt19937 generator{rd()};

    //! file descriptor of the opened file
    int fd{-1};

    //! name of the output file generated
    std::string filename;

    //! size of the ASCII header in bytes
    uint32_t header_size = 4096;

    std::array<int8_t, 32> known_input_data {
      -12, 12, -5, -1, -3, 1, -2, 2, 0, -7,
      1, -2, 0, -0, 2, 4, 0, -3, 1, -1,
      -1, -4, -1, -3, -2, 2, -6, 1, -2, 2,
      0, 1
    };
  };

} // namespace dsp::test

#endif // __dsp_DADAOutputFileTest_h
