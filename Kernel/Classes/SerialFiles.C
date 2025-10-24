/***************************************************************************
 *
 *   Copyright (C) 2002 - 2023 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SerialFiles.h"

#include "Error.h"
#include "templates.h"
#include "dirutil.h"
#include "strutil.h"

#include <algorithm>
#include <math.h>

using namespace std;

dsp::SerialFiles::SerialFiles (const char* name) : MultiFile (name)
{
  test_contiguity = true;
  current_index = 0;
}

dsp::SerialFiles::~SerialFiles ()
{
}

void dsp::SerialFiles::force_contiguity ()
{
  test_contiguity = false;
}

void dsp::SerialFiles::after_open (File* loader)
{
  loader->close();
}

void dsp::SerialFiles::setup () try
{
  if (test_contiguity)
    ensure_contiguity();

  if (verbose)
    cerr << "dsp::SerialFiles::setup new Observation" << endl;

  info = files[0]->get_info()->clone();

  if (verbose)
    cerr << "dsp::SerialFiles::setup count total samples" << endl;

  uint64_t total_ndat = 0;
  for( unsigned i=0; i<files.size(); i++)
    total_ndat += files[i]->get_info()->get_ndat();

  get_info()->set_ndat (total_ndat);

  loader = files[0];

  if (verbose)
    cerr << "dsp::SerialFiles::setup call loader reopen" << endl;
  loader->reopen();

  // SerialFiles must reflect the time sample resolution of the underlying device
  resolution = loader->resolution;

  current_index = 0;
  current_filename = files[0]->get_filename();

  rewind ();
}
catch (Error& err)
{
  throw err += "dsp::SerialFiles::setup";
}

//! Makes sure only these filenames are open
void dsp::SerialFiles::have_open (const vector<string>& filenames)
{
  // Erase any files we already have open that we don't want open
  for (unsigned ifile=0; ifile<files.size(); ifile++)
  {
    if ( !found(files[ifile]->get_filename(),filenames) )
    {
      files.erase(files.begin()+ifile);
      ifile--;
    }
  }

  open (filenames);
}

//! Erase the entire list of loadable files
void dsp::SerialFiles::erase_files()
{
  files.erase( files.begin(), files.end());
  loader = 0;
  info = 0;
  rewind ();
}

//! Erase just some of the list of loadable files
void dsp::SerialFiles::erase_files(const vector<string>& erase_filenames)
{
  for( unsigned ifile=0; ifile<files.size(); ifile++)
  {
    if( found(files[ifile]->get_filename(),erase_filenames) )
    {
      files.erase( files.begin()+ifile );
      ifile--;
    }
  }
  
  if( files.empty() )
  {
    erase_files ();
    return;
  }

  if (test_contiguity)
    ensure_contiguity();

  setup();
}

bool time_order (const dsp::File* a, const dsp::File* b)
{
  return a->get_info()->get_start_time() < b->get_info()->get_start_time();
}

void dsp::SerialFiles::ensure_contiguity() try
{
  if (verbose)
    cerr << "dsp::SerialFiles::ensure_contiguity enter" << endl;

  sort( files.begin(), files.end(), time_order );

  for (unsigned ifile=1; ifile<files.size(); ifile++)
  {
    if (verbose)
      cerr << "dsp::SerialFiles::ensure_contiguity files " << ifile-1 
	   << " and " << ifile << endl;

    File* file1 = files[ifile-1];
    File* file2 = files[ifile];

    if ( !file1->contiguous(file2) )
          throw Error (InvalidParam, "dsp::Multifile::ensure_contiguity",
                   "file %d (%s)\n\tis not contiguous with\n\tfile %d (%s)",
                   ifile-1, file1->get_filename().c_str(),
                   ifile, file2->get_filename().c_str());
  }

  if (verbose)
    cerr << "dsp::SerialFiles::ensure_contiguity return" << endl;
}
catch (Error& err)
{
  throw err += "dsp::SerialFiles::ensure_contiguity";
}

//! Load bytes from file
int64_t dsp::SerialFiles::load_bytes (unsigned char* buffer, uint64_t bytes) try 
{
  if (verbose)
    cerr << "dsp::SerialFiles::load_bytes nbytes=" << bytes << endl;
  
  if (!loader)
    throw Error(InvalidState,"dsp::SerialFiles::load_bytes",
		"No loader.  Possible SerialFiles::open failure.");

  loader->set_output( get_output() );

  uint64_t bytes_loaded = 0;
  unsigned index = current_index;

  while (bytes_loaded < bytes)
  {
    int64_t to_load = bytes - bytes_loaded;

    if (index >= files.size())
    {
      if (verbose)
        cerr << "dsp::SerialFiles::load_bytes end of data" << endl;
      end_of_data = true;
      break;
    }

    // Ensure we are loading from correct file
    set_loader (index);

    if (verbose)
      cerr << "dsp::SerialFiles::load_bytes calling loader load_bytes (" << to_load << ")" << endl;

    int64_t did_load = loader->load_bytes (buffer, to_load);

    if (verbose)
      cerr << "dsp::SerialFiles::load_bytes loaded " << did_load << " bytes" << endl;

    if (did_load < 0)
      return -1;

    if (did_load < to_load)
      // this File has reached the end of data
      index ++;

    bytes_loaded += did_load;
    buffer += did_load;
  }

  return bytes_loaded;
}
catch (Error& err)
{
  throw err += "dsp::SerialFiles::load_bytes";
}

//! Adjust the file pointer
int64_t dsp::SerialFiles::seek_bytes (uint64_t bytes)
{
  if( !loader )
    throw Error(InvalidState, "dsp::SerialFiles::seek_bytes",
		"no loader.  Have you called SerialFiles::open() yet?");

  if (verbose)
    cerr << "dsp::SerialFiles::seek_bytes nbytes=" << bytes << endl;

  // Total number of bytes stored in files thus far
  uint64_t total_bytes = 0;

  unsigned index;
  for (index = 0; index < files.size(); index++)
  {
    // Number of bytes stored in this file
    uint64_t file_bytes = files[index]->get_info()->get_nbytes();

    if (verbose)
      cerr << "dsp::SerialFiles::seek_bytes file[" << index << "] nbytes=" << file_bytes << endl;

    if (bytes < total_bytes + file_bytes)
      break;

    total_bytes += file_bytes;
  }

  if (index == files.size())
  {
    cerr << "dsp::SerialFiles::seek_bytes (" << bytes << ") past end of data" << endl;
    return -1;
  }

  if (verbose)
    cerr << "dsp::SerialFiles::seek_bytes index=" << index << endl;

  set_loader (index);

  if (verbose)
    cerr << "dsp::SerialFiles::seek_bytes calling loader seek_bytes (" << bytes-total_bytes << ")" << endl;

  int64_t seeked = loader->seek_bytes (bytes-total_bytes);
  if (seeked < 0)
    return -1;

  return total_bytes + seeked;
}

void dsp::SerialFiles::set_loader (unsigned index) try
{
  if (verbose)
    cerr << "dsp::SerialFiles::set_loader index=" << index << " current=" << current_index << endl;

  if (index == current_index)
    return;

  // Close previously open file
  if (loader)
  {
    if (verbose)
      cerr << "dsp::MultiFile::set_loader calling Input::close" << endl;
    loader->close();
  }

  loader = files[index];

  loader->set_output( get_output() );

  if (verbose)
    cerr << "dsp::MultiFile::set_loader calling Input::reopen" << endl;

  loader->reopen();

  uint64_t preceding_ndat = 0;
  for( unsigned i=0; i<index; i++)
    preceding_ndat += files[i]->get_info()->get_ndat();

  loader->start_offset = start_offset - preceding_ndat;

  if (verbose)
    cerr << "dsp::MultiFile::set_loader this->start_offset=" << start_offset << 
      " preceding_ndat=" << preceding_ndat << 
      " loader->start_offset=" << loader->start_offset << endl;

  current_index = index;
  current_filename = files[index]->get_filename();
}
catch (Error& err)
{
  throw err += "dsp::SerialFiles::set_loader";
}

void dsp::SerialFiles::add_extensions (Extensions *ext)
{
  if (loader)
    loader->add_extensions(ext);
}

bool dsp::SerialFiles::has_loader ()
{
  return loader;
}

dsp::File* dsp::SerialFiles::get_loader ()
{
  return loader;
}

const dsp::File* dsp::SerialFiles::get_loader () const
{
  return loader;
}

