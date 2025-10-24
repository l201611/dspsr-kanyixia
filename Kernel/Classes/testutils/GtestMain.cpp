/***************************************************************************
 *
 *   Copyright (C) 2024 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/GtestMain.h"

// for access to verbosity flags
#include "dsp/Operation.h"
#include "dsp/Observation.h"
#include "MJD.h"

#include <unistd.h>
#include <sstream>
#include <fstream>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace dsp::test {

auto test_data_dir() -> std::string&
{
  static std::string data_dir = ".";
  return data_dir;
}

auto test_data_file(std::string const& filename) -> std::string
{
  return test_data_dir() + "/" + filename;
}

std::vector<bool> get_gpu_flags()
{
#ifdef HAVE_CUDA
  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

  if (cudaStatus == cudaSuccess && deviceCount > 0)
  {
    return {false, true};
  }

  std::cout << "No GPU detected ... tests of CUDA disabled" << std::endl;
#endif
  return {false};
}

auto gtest_main(int argc, char** argv) -> int
{
  // will process gtest options and pass on the rest
  ::testing::InitGoogleTest(&argc, argv);

  // turn off all verbose logs by default, re-enable when assessing code coverage/correctness
  dsp::Observation::verbose = false;
  dsp::Operation::verbose = false;
  MJD::verbose = false;

  // process extra command line options;
  for (int i=0; i < argc; i++)
  {
    std::string const arg(argv[i]); // NOLINT
    if (arg == "--test_data")
    {
      if(++i < argc)
      {
        std::string const val(argv[i]); //NOLINT
        test_data_dir() = val;
      }
    }
    else if (arg == "--info")
    {
      dsp::Observation::verbose = false;
      dsp::Operation::verbose = false;
      MJD::verbose = false;
    }
    else if (arg == "--debug")
    {
      dsp::Operation::verbose = true;
    }
    else if (arg == "--trace")
    {
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;
      MJD::verbose = true;
    }
    else if (arg == "--warn")
    {
      dsp::Observation::verbose = false;
      dsp::Operation::verbose = false;
      MJD::verbose = false;
    }
  }

  return RUN_ALL_TESTS();
}

} // namespace dsp::test
