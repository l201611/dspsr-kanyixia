/***************************************************************************
 *
 *   Copyright (C) 2024 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <gtest/gtest.h>
#include <string>

#ifndef __dsp_GtestMain_h
#define __dsp_GtestMain_h

namespace dsp::test {

/*
 * @brief the data directory to find test data files
 */
std::string& test_data_dir();

/*
 * @brief return the filename with the test_data_dir prepended
 */
std::string test_data_file(std::string const& filename);

/**
 * @brief Executable function to launch gtests
 */
int gtest_main(int argc, char** argv);

/**
 * @brief return {false,true} if GPU is available; otherwise return {false}
 */
std::vector<bool> get_gpu_flags();

} // namespace dsp::test

#endif // __dsp_GtestMain_h
