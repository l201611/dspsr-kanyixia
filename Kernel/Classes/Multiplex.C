//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Jonathon Kocz
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Multiplex.h"

#include "Error.h"
#include "templates.h"
#include "dirutil.h"
#include "strutil.h"

#include <algorithm>
#include <math.h>

using namespace std;

dsp::Multiplex::Multiplex () : MultiFile ( "Multiplex" )
{
  //current_index = 0;
}

dsp::Multiplex::~Multiplex ()
{
}

int64_t dsp::Multiplex::load_bytes (unsigned char* buffer, uint64_t bytes)
{
  if (verbose)
    cerr << "Multiplex::load_bytes nbytes=" << bytes << endl;
  
  if( !loader )
    throw Error(InvalidState,"dsp::Multiplex::load_bytes",
		"No loader.  Possible Multiplex::open failure.");

  uint64_t bytes_loaded = 0;
  uint64_t to_load = 0;
  uint64_t packet_size = 8192;
  uint64_t end_of_packet = 0;
  uint64_t byte_offset = 0;
  uint64_t load_index = current_index;
  bool increment_index = false;

  //cerr << "current index " << current_index << endl;

  while (bytes_loaded < bytes)
  {
    to_load = bytes - bytes_loaded;
    end_of_packet = packet_size - byte_offset;

    if (to_load > end_of_packet) 
    {
      //end_of_packet = packet size - byte_offset;
      to_load = end_of_packet;
      increment_index = true;
    }
    int64_t did_load = loader->load_bytes (buffer, to_load);

    if (did_load < 0)
      return -1;
	
    if (did_load < int64_t(to_load))
    {
      end_of_data = true;
      if (verbose)
        cerr << "dsp::Multiplex::end_of_data" << endl;
      break;
    }

    //  if (did_load < to_load)
      // this File has reached the end of data
    // index ++;

    if (increment_index)
    {
      load_index++;
      if (load_index >= files.size())
        load_index = 0;

      //cerr << "next load_index " << load_index << endl;
      //cerr << "files.size " << files.size() << endl;
      loader = files[load_index];

      byte_offset = 0;
      increment_index = false;
    }
    else
    {
      byte_offset += to_load;
    }
    if (load_index >= files.size())
    {
      load_index = 0;
      if (verbose)
        cerr << "dsp::Multiplex::load_bytes recycling load_index" << endl;
    }
    bytes_loaded += did_load;
    buffer += did_load;
  }
  current_index = load_index;
  return bytes_loaded;
}

int64_t dsp::Multiplex::seek_bytes (uint64_t bytes)
{
  if( !loader )
    throw Error(InvalidState, "dsp::Multiplex::seek_bytes",
      "no loader.  Have you called Multiplex::open() yet?");

  if (verbose)
    cerr << "Multiplex::seek_bytes nbytes=" << bytes << endl;

  // Total number of bytes stored in files thus far
  uint64_t total_bytes = 0;
  uint64_t packet_size = 8192;
  uint64_t packet_counter = 0;
 
  unsigned index = 0;

  for(index = 0; index < files.size(); index++)
  {
    total_bytes += files[index]->get_info()->get_nbytes();
  }

  uint64_t file_bytes = 0;
  
  while ((bytes > file_bytes + packet_size) && (total_bytes > file_bytes))
  {
    file_bytes += packet_size;
    index++;

    if (index == files.size())
    {
      index = 0;
      packet_counter++;
    }
  }

  if (total_bytes < file_bytes)
  {
    cerr << "dsp::Multiplex::seek_bytes (" << bytes << ")"
      " past end of data" << endl;
    return -1;
  }

  int64_t seeked = loader->seek_bytes (bytes - file_bytes);
  if (seeked < 0)
    return -1;
  
  return file_bytes + seeked;
}
