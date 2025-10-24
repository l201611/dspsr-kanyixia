/***************************************************************************
 *
 *   Copyright (C) 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SingleFile.h"
#include "dsp/ParallelBitSeries.h"

#include "dsp/File.h"

#include "Expect.h"
#include "strutil.h"

using namespace std;

dsp::SingleFile::SingleFile () : ParallelInput("SingleFile") {}

dsp::SingleFile::~SingleFile () {}

std::string extract_filename (const std::string& descriptor)
{
  Expect test (descriptor.c_str());

  static const char* expect = "SINGLE FILE:";
  if (!test.expect(expect))
    throw Error (InvalidParam, "dsp::SingleFile::matches", "first characters != '%s'", expect);

  std::vector<char> filename(FILENAME_MAX);
  if (fgets (filename.data(), FILENAME_MAX, test.fptr()) == NULL)
    throw Error (InvalidParam, "dsp::SingleFile::matches", "fgets filename");

  string whitespace = " \t\n";
  string fname = filename.data();
  return stringtok(fname, whitespace);
}

bool dsp::SingleFile::matches (const std::string& descriptor) const try
{
  // cerr << "dsp::SingleFile::matches descriptor=" << descriptor << endl;
  std::string filename = extract_filename (descriptor);

  // cerr << "dsp::SingleFile::matches try opening " << filename << endl;
  Reference::To<File> file = File::create(filename.c_str());
  return true;
}
catch (Error& error)
{
  // cerr << "dsp::SingleFile::matches error " << error << endl;
  return false;
}

void dsp::SingleFile::open (const std::string& descriptor) try
{
  std::string filename = extract_filename (descriptor);
  // cerr << "dsp::SingleFile::open opening " << filename << endl;

  Reference::To<Input> file = File::create(filename.c_str());
  inputs.resize(1);
  inputs[0] = file;

  info = file->get_info();
}
catch (Error& error)
{
  // cerr << "dsp::SingleFile::open error " << error << endl;
  throw error;
}

void dsp::SingleFile::load (ParallelBitSeries* bitseries)
{
  if (verbose)
    cerr << "dsp::SingleFile::load" << endl;

  ParallelInput::load(bitseries);
  bitseries->copy_configuration(bitseries->at(0));
}

//! Return the number of time samples to load on each load_block
uint64_t dsp::SingleFile::get_block_size () const
{
  return inputs.at(0)->get_block_size();
}

//! Set the number of time samples to load on each load_block
void dsp::SingleFile::set_block_size (uint64_t _size)
{
  inputs.at(0)->set_block_size(_size);
}

//! Return the number of time samples by which consecutive blocks overlap
uint64_t dsp::SingleFile::get_overlap () const
{
  return inputs.at(0)->get_overlap();
}

//! Set the number of time samples by which consecutive blocks overlap
void dsp::SingleFile::set_overlap (uint64_t _overlap)
{
  // cerr << "dsp::SingleFile::set_overlap " << _overlap << " bytes" << endl;
  inputs.at(0)->set_overlap(_overlap);
}

//! Return the number of time samples by which consecutive blocks overlap
unsigned dsp::SingleFile::get_resolution () const
{
  return inputs.at(0)->get_resolution();
}
