/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ASCIIObservation.h"
#include "dsp/DualFile.h"
#include "dsp/File.h"

#include "relative_path.h"
#include "Error.h"

#include <fstream>

using namespace std;

dsp::DualFile::DualFile () : ParallelInput("DualFile") { }

dsp::DualFile::~DualFile () { }

//! interpret the descriptor as a filename and return a pair of descriptors for the DualFile input
std::pair<std::string, std::string> read_metadata_file (const std::string& descriptor)
{
  std::ifstream metadata_file(descriptor);
  if (!metadata_file.good())
  {
    metadata_file.close();
    throw Error (FailedSys, "dsp::DualFile::read_metadata_file", "failed to open file: %s", descriptor.c_str());
  }

  std::string expect = "DUAL FILE:";
  std::string first_line;
  if (std::getline(metadata_file, first_line, '\n'))
  {
    if (first_line.compare(expect) != 0)
    {
      metadata_file.close();
      throw Error (InvalidParam, "dsp::DualFile::read_metadata_file", "first line was not '%s'", expect.c_str());
    }
  }
  else
  {
    metadata_file.close();
    throw Error (InvalidParam, "dsp::DualFile::read_metadata_file", "file was empty");
  }

  std::pair<std::string, std::string> descriptors;

  if (std::getline(metadata_file, descriptors.first, '\n'))
  {
    relative_path (descriptor, descriptors.first);
    if (! dsp::File::can_create(descriptors.first.c_str()))
      throw Error(InvalidParam, "dsp::DualFile::read_metadata_file", "%s not a valid File", descriptors.first.c_str());
  }
  else
  {
    throw Error (InvalidParam, "dsp::DualFile::read_metadata_file", "could not read first descriptor");
  }

  if (std::getline(metadata_file, descriptors.second, '\n'))
  {
    relative_path (descriptor, descriptors.second);
    if (!dsp::File::can_create(descriptors.second.c_str()))
      throw Error(InvalidParam, "dsp::DualFile::read_metadata_file", "%s not a valid File", descriptors.second.c_str());
  }
  else
  {
    throw Error (InvalidParam, "dsp::DualFile::read_metadata_file", "could not read second descriptor");
  }

  metadata_file.close();
  return descriptors;
}

bool dsp::DualFile::matches (const std::string& descriptor) const try
{
  if (verbose)
    cerr << "dsp::DualFile::matches descriptor=" << descriptor << endl;
  read_metadata_file(descriptor);
  return true;
}
catch (Error& error)
{
  cerr << "dsp::DualFile::matches error " << error << endl;
  return false;
}

void dsp::DualFile::open (const std::string& descriptor) try
{
  if (verbose)
    cerr << "dsp::DualFile::open descriptor=" << descriptor << endl;
  std::pair<std::string, std::string> descriptors = read_metadata_file(descriptor);

  if (verbose)
    cerr << "dsp::DualFile::open creating Files from " << descriptors.first << " and " << descriptors.second << endl;
  Reference::To<dsp::File> first = dsp::File::create(descriptors.first);
  Reference::To<dsp::File> second = dsp::File::create(descriptors.second);

  set_inputs(first,second);
}
catch (Error& error)
{
  cerr << "dsp::DualFile::open error " << error << endl;
  throw error;
}

void dsp::DualFile::set_inputs (Input* first, Input* second)
{
  inputs.resize(2);
  inputs[0] = first;
  inputs[1] = second;

  info = inputs[0]->get_info();
}

void dsp::DualFile::load (ParallelBitSeries* bitseries)
{
  if (verbose)
    cerr << "dsp::DualFile::load bitseries=" << bitseries << endl;

  ParallelInput::load(bitseries);
  bitseries->copy_configuration(bitseries->at(0));
}

uint64_t dsp::DualFile::get_block_size () const
{
  return inputs.at(0)->get_block_size();
}

void dsp::DualFile::set_block_size (uint64_t _size)
{
  if (verbose)
    cerr << "dsp::DualFile::set_block_size size=" << _size << endl;

  uint64_t res_0 = inputs.at(0)->get_resolution();
  uint64_t res_1 = inputs.at(1)->get_resolution();

  if (res_0 == res_1)
  {
    if (verbose)
      cerr << "dsp::DualFile::set_block_size input[0] and input[1] have resolution=" << res_0 << endl;

    inputs.at(0)->set_block_size(_size);
    inputs.at(1)->set_block_size(_size);
  }
  else
  {
    if (verbose)
      cerr << "dsp::DualFile::set_block_size input[0].resolution=" << res_0
           << " input[1].resolution=" << res_1 << endl;

    uint64_t res_blocks = _size / res_0;
    if (verbose && _size % res_0)
      cerr << "dsp::DualFile::set_block_size size=" << _size << " is not divisible by resolution=" << res_0 << endl;
    _size = res_blocks * res_0;

    inputs.at(0)->set_block_size(_size);
    uint64_t other_size = res_blocks * res_1;
    inputs.at(1)->set_block_size(other_size);
  }
}

uint64_t dsp::DualFile::get_overlap () const
{
  return inputs.at(0)->get_overlap();
}

void dsp::DualFile::set_overlap (uint64_t _overlap)
{
  inputs.at(0)->set_overlap(_overlap);
}

unsigned dsp::DualFile::get_resolution () const
{
  return inputs.at(0)->get_resolution();
}
