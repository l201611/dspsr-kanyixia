/***************************************************************************
 *
 *   Copyright (C) 2005 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "environ.h"
#include "dsp/BlockFile.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

//! Constructor
dsp::BlockFile::BlockFile (const char* name) : File (name)
{
  // Total number of bytes in a block
  block_bytes = 0;

  block_header_bytes = 0;
  block_tailer_bytes = 0;
  current_block_byte = 0;

  total_bytes_missing = 0;
}

//! Destructor
dsp::BlockFile::~BlockFile ()
{
}

uint64_t dsp::BlockFile::get_block_data_bytes() const
{
  if (!block_bytes)
    throw Error (InvalidState, "dsp::BlockFile::get_block_data_bytes",
		 "undefined block size");

  uint64_t non_data_bytes = block_header_bytes + block_tailer_bytes;

  if (non_data_bytes > block_bytes)
    throw Error (InvalidState, "dsp::BlockFile::get_block_data_bytes",
		 "block_bytes=" UI64 " < header+tailer_bytes=" UI64,
		 block_bytes, non_data_bytes);

  return block_bytes - non_data_bytes;
}

//! Load bytes from file
//TODO
// memset(...)
// TEST
int64_t dsp::BlockFile::load_bytes (unsigned char* buffer, uint64_t bytes) try
{
  uint64_t block_data_bytes = get_block_data_bytes ();
  uint64_t to_load = bytes;

  if (verbose)
  {
    cerr << "dsp::BlockFile::load_bytes nbytes=" << bytes << " missing=" << total_bytes_missing << endl;
    off_t current_offset = lseek (fd, 0, SEEK_CUR);
    cerr << "dsp::BlockFile::load_bytes current_offset=" << current_offset << endl;
  }

  // Get real data
  while (to_load)
  {
    ssize_t bytes_read = 0;

    if (total_bytes_missing > 0)
    {
      // to_read = number of bytes to read
      uint64_t to_zero = std::min (to_load, total_bytes_missing);

      // buffer = where to write in memory
      // zeros = data to be written
      // to_read = number of bytes of 'zeros' to be written
      // memset() returns a pointer to memory area of 'buffer'
      memset(buffer, 0, to_zero);
      bytes_read = to_zero;
      total_bytes_missing -= to_zero;
    }
    else
    {
      // to_read = number of bytes to read
      uint64_t to_read = std::min (to_load, block_data_bytes - current_block_byte);

      if (verbose)
      {
        cerr << "dsp::BlockFile::load_bytes block_data_bytes=" << block_data_bytes
             << " current_block_byte=" << current_block_byte << endl;
        cerr << "dsp::BlockFile::load_bytes to_read=" << to_read << endl;
      }

      // read(a,b,c) reads 'c'-numOfBytes from 'a', 'filedes', to location 'b' and returns the number of bytes it read.
      bytes_read = read(fd, buffer, to_read);

      if (bytes_read < 0)
      {
        throw Error (FailedSys, "dsp::BlockFile::load_bytes", "read(%d)", fd);
      }

      current_block_byte += bytes_read;

      // probably the end of file
      if (uint64_t(bytes_read) < to_read)
      {
        if (verbose)
          cerr << "dsp::BlockFile::load_bytes bytes_read=" << bytes_read << " < " << " to_read=" << to_read << endl;
	break;
      }
    }

    // decrement to_load by the number of bytes that were read (bytes_read)
    to_load -= bytes_read;
    // increment buffer and current_block_byte by same amount (bytes_read)
    buffer += bytes_read;

    if (current_block_byte == block_data_bytes)
    {
      current_block_byte = 0;

      // get missing number of packets, if any
      uint64_t missed_packets = skip_extra();

      total_bytes_missing = missed_packets * block_data_bytes;
    }
  }

  // should return original size of bytes(?)
  return bytes - to_load;
}
catch (Error& error)
{
  throw error += "dsp::BlockFile::load_bytes";
}

/*! Derived types should calculate and return number of missed packets */
uint64_t dsp::BlockFile::skip_extra ()
{
  if (verbose)
    cerr << "BlockFile::skip_extra skip one tailer and header" << endl;

  if (lseek (fd, block_header_bytes + block_tailer_bytes, SEEK_CUR) < 0)
    throw Error (FailedSys, "dsp::BlockFile::load_bytes", "seek(%d)", fd);

  return 0;
}

//! Adjust the file pointer
int64_t dsp::BlockFile::seek_bytes (uint64_t nbytes)
{
  if (verbose)
    cerr << "dsp::BlockFile::seek_bytes nbytes=" << nbytes << endl;

  if (fd < 0)
    throw Error (InvalidState, "dsp::BlockFile::seek_bytes", "invalid fd");

  if (total_bytes_missing)
    throw Error (InvalidState, "dsp::BlockFile::seek_bytes",
		 "seeking through missing blocks not implemented");

  uint64_t block_data_bytes = get_block_data_bytes ();

  if (verbose)
    cerr << "dsp::BlockFile::seek_bytes block_bytes=" << block_bytes
         << " block_header_bytes=" << block_header_bytes
         << " block_tailer_bytes=" << block_tailer_bytes << endl;

  uint64_t current_block = nbytes / block_data_bytes;
  current_block_byte = nbytes % block_data_bytes;

  if (verbose)
    cerr << "dsp::BlockFile::seek_bytes current_block="<< current_block <<endl;

  uint64_t tot_header_bytes = (current_block+1) * block_header_bytes;
  uint64_t tot_tailer_bytes = current_block * block_tailer_bytes;

  uint64_t to_byte = nbytes + header_bytes + tot_header_bytes + tot_tailer_bytes;

  if (verbose)
    cerr << "dsp::BlockFile::seek_bytes SEEK_SET to " << to_byte << endl;

  if (lseek (fd, to_byte, SEEK_SET) < 0)
    throw Error (FailedSys, "dsp::BlockFile::seek_bytes",
		 "lseek (" UI64 ")", to_byte);

  return nbytes;
}

int64_t dsp::BlockFile::fstat_file_ndat (uint64_t tailer_bytes)
{
  if (verbose)
    cerr << "dsp::BlockFile::fstat_file_ndat header=" << header_bytes
	 << " block=" << block_bytes
	 << " data=" << get_block_data_bytes() << endl;

  struct stat file_stats;

  if (fstat(fd, &file_stats) != 0)
    throw Error (FailedCall, "dsp::BlockFile::fstat_file_ndat","fstat(%d)",fd);

  int64_t actual_file_sz = file_stats.st_size - header_bytes - tailer_bytes;

  uint64_t nblocks = actual_file_sz / block_bytes;
  uint64_t extra = actual_file_sz % block_bytes;

  uint64_t block_data_bytes = get_block_data_bytes ();

  uint64_t data_bytes = nblocks * block_data_bytes;

  if (extra > block_header_bytes) {
    extra -= block_header_bytes;
    if (extra > block_data_bytes)
      extra = block_data_bytes;
    data_bytes += extra;
  }

  return get_info()->get_nsamples (data_bytes);
}
