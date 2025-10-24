/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/OutputFile.h"
#include "dsp/BitSeries.h"

#include "Error.h"

#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

using namespace std;

//! Constructor
dsp::OutputFile::OutputFile (const char* operation_name)
  : Sink<BitSeries> (operation_name)
{
}

//! Destructor
dsp::OutputFile::~OutputFile ()
{
  close();
}

void dsp::OutputFile::close()
{
  if (fd != -1)
    ::close (fd);

  fd = -1;
  output_filename = "";
}

void dsp::OutputFile::set_fractional_second_decimal_places(unsigned digits)
{
  fractional_second_decimal_places = digits;
}

bool dsp::OutputFile::input_is_valid() const
{
  return input->get_rate() > 0.0 && input->get_centre_frequency() != 0.0 && input->get_bandwidth() != 0.0;
}

void dsp::OutputFile::calculation ()
{
  if (fd == -1)
  {
    if (!input_is_valid())
    {
      if (verbose)
        cerr << "dsp::OutputFile::calculation refusing to output invalid input" << endl;
      return;
    }

    if (output_filename.empty())
    {
      MJD epoch = input->get_start_time();
      string filename = epoch.datestr( datestr_pattern.c_str(), 13 );
      output_filename = filename + get_extension();
    }

    open_file (output_filename);
  }

  unload_bytes (input->get_rawptr(), input->get_nbytes());
}

void dsp::OutputFile::open_file (const char* filename)
{
  int oflag = O_WRONLY | O_CREAT | O_TRUNC | O_EXCL;
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP;

  fd = ::open (filename, oflag, mode);
  if (fd < 0)
    throw Error (FailedSys, "dsp::OutputFile::open_file",
		 "error open("+output_filename+")");

  write_header ();
    
  header_bytes = lseek(fd,0,SEEK_CUR);
}

//! Load nbyte bytes of sampled data from the device into buffer
int64_t dsp::OutputFile::unload_bytes (const void* buffer, uint64_t nbyte)
{
  int64_t written = ::write (fd, buffer, nbyte);

  if (written < (int64_t) nbyte)
  {
    Error error (FailedSys, "dsp::OutputFile::unload_bytes");
    error << "error write(fd=" << fd << ",buf=" << buffer
	  << ",nbyte=" << nbyte;
    throw error;
  }

  return written;
}
