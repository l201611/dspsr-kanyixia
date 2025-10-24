/***************************************************************************
 *
 *   Copyright (C) 2002 - 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/DADABuffer.h"
#include "dsp/ASCIIObservation.h"
#include "dsp/BitSeries.h"

#include "ascii_header.h"
#include "FilePtr.h"

#if HAVE_CUDA
#include "dada_cuda.h"
#include "ipcio_cuda.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <fstream>
using namespace std;

//! Constructor
dsp::DADABuffer::DADABuffer ()
  : File ("DADABuffer")
{
  /*
    when a block overlap policy is necessary (e.g. when using two GPUs)
    different threads cannot recycle BitSeries data and must share a
    single overlap buffer.
  */
  set_overlap_buffer(new BitSeries);
}

dsp::DADABuffer::~DADABuffer ()
{
  disconnect ();
#if HAVE_CUDA
  if (zeroed_buffer)
  {
    cudaFree (zeroed_buffer);
    zeroed_buffer = NULL;
    zeroed_buffer_size = 0;
  }
#endif
}

void dsp::DADABuffer::close ()
{
  if (verbose)
    cerr << "dsp::DADABuffer::close start" << endl;

  if (!hdu)
    return;

  if (ipcio_is_open(hdu->data_block))
  {
    if (verbose)
      cerr << "dsp::DADABuffer::close call ipcio_close" << endl;

    if (ipcio_close(hdu->data_block) < 0)
    {
      throw Error (FailedCall, "dsp::DADABuffer::close", "failed call to ipcio_close");
    }
  }
}

void dsp::DADABuffer::disconnect ()
{
  if (verbose)
    cerr << "dsp::DADABuffer::disconnect start" << endl;

  if (!hdu)
    return;

  if (!eod())
  {
    if (verbose)
      cerr << "dsp::DADABuffer::disconnect NOT eod" << endl;
    /*
       If the stream has not reached end of data, then do not mark the
       header as cleared.  The next few lines work with the implementation
       of dada_hdu_unlock_read and should probably be given a less
       obscure interface.  By not clearing the header, the script that
       runs dspsr can use dbnull to clear the ring buffer of the data
       that caused dspsr to quit before end-of-data.
    */
    if (hdu->header)
    {
      free (hdu->header);
      hdu->header = 0;
    }
  }

#if HAVE_CUDA
  if (!passive && registered_memory)
  {
    if (dada_cuda_dbunregister (hdu) < 0)
    {
      throw Error (FailedCall, "dsp::DADABuffer::disconnect",
        "cannot unregister ring buffer blocks as Pinned memory");
    }
  }
#endif
  if (!passive && dada_hdu_unlock_read (hdu) < 0)
    cerr << "dsp::DADABuffer::disconnect error during dada_hdu_unlock_read" << endl;

  if (passive && dada_hdu_close_view (hdu) < 0)
    cerr << "dsp::DADABuffer::disconnect error during dada_hdu_close_view" << endl;

  if (dada_hdu_disconnect (hdu) < 0)
    cerr << "dsp::DADABuffer::disconnect error during dada_hdu_disconnect" << endl;

  dada_hdu_destroy (hdu);

  if (verbose)
    cerr << "dsp::DADABuffer::disconnect return" << endl;

  hdu = 0;
}

void dsp::DADABuffer::rewind ()
{
  if (verbose)
    cerr << "dsp::DADABuffer::rewind start" << endl;

  end_of_data = false;
  current_sample = 0;

  if (!passive)
    seek (0,SEEK_SET);

  if (verbose)
    cerr << "dsp::DADABuffer::rewind return" << endl;

  last_load_ndat = 0;
}

bool dsp::DADABuffer::is_valid (const char* filename) const
{
  FilePtr ptr = fopen (filename, "r");
  if (!ptr)
    return false;

  char first[16];
  if (fgets (first, 16, ptr) == NULL)
    return false;

  first[10] = '\0';

  char expect[16] = "DADA INFO:";

  if (strcmp (first, expect) == 0)
    return true;

  if (verbose)
    cerr << "dsp::DADABuffer::is_valid first 10 characters '" << first << "'"
            " != '" << expect << "'" << endl;

  return false;
}

int ipcio_view_eod (ipcio_t* ipcio, unsigned byte_resolution)
{
  ipcbuf_t* buf = &(ipcio->buf);

#ifdef _DEBUG
  fprintf (stderr, "ipcio_view_eod: write_buf=%"PRIu64"\n",
	   ipcbuf_get_write_count( buf ) );
#endif

  buf->viewbuf ++;

  if (ipcbuf_get_write_count( buf ) > buf->viewbuf)
    buf->viewbuf = ipcbuf_get_write_count( buf ) + 1;

  ipcio->bytes = 0;
  ipcio->curbuf = 0;

  uint64_t current = ipcio_tell (ipcio);
  uint64_t too_far = current % byte_resolution;
  if (too_far)
  {
    int64_t absolute_bytes = ipcio_seek (ipcio,
					 current + byte_resolution - too_far,
					 SEEK_SET);
    if (absolute_bytes < 0)
      return -1;
  }

  return 0;
}

void dsp::DADABuffer::open_file (const char* filename)
{
  if (verbose)
    cerr << "dsp::DADABuffer::open_file" << endl;

  ifstream input (filename);
  if (!input)
    throw Error (FailedCall, "dsp::DADABuffer::open_file",
		 "cannot open INFO file: %s", filename);

  std::string line;
  std::getline (input, line);

  if (line != "DADA INFO:")
    throw Error (InvalidParam, "dsp::DADABuffer::open_file",
		 "invalid INFO file (no preamble): %s", filename);

  input >> line;

  if (line != "key")
    throw Error (InvalidParam, "dsp::DADABuffer::open_file",
		 "invalid INFO file (no key): %s", filename);

  input >> line;
  string key = line;
  input >> line;
  passive = line == "viewer";

  if (verbose)
    cerr << "dsp::DADABuffer::open_file key=" << key << " passive=" << passive << endl;

  open_key(key);
  reopen();
}

void dsp::DADABuffer::open_key (const string& text)
{
  key_t key = 0;
  int scanned = 0;

  if (text.length())
    scanned = sscanf (text.c_str(), "%x", &key);

  if (scanned != 1)
    throw Error (InvalidParam, "dsp::DADABuffer::open_key", "invalid key='"+text+"'");

  if (!hdu)
  {
    /* DADA library logger */
    multilog_t* log = multilog_open ("dspsr", 0);
    multilog_add (log, stderr);

    hdu = dada_hdu_create (log);
  }

  assert(hdu != nullptr);

  dada_hdu_set_key (hdu, key);

  if (dada_hdu_connect (hdu) < 0)
  {
    dada_hdu_destroy (hdu);
    hdu = 0;
    throw Error (FailedCall, "dsp::DADABuffer::open_key", "cannot connect to DADA ring buffers");
  }

  if (!passive && dada_hdu_lock_read (hdu) < 0)
  {
    dada_hdu_disconnect (hdu);
    dada_hdu_destroy (hdu);
    hdu = 0;
    throw Error (FailedCall, "dsp::DADABuffer::open_key",
    "cannot lock DADA ring buffer read client status");
  }

#if HAVE_CUDA
  if (!passive)
  {
    if (verbose)
      cerr << "dsp::DADABuffer::open_key registering dada buffers with CUDA for pinned transfers" << endl;
    // attempt to register the shared memory with the CUDA driver
    registered_memory = (dada_cuda_dbregister (hdu) == 0);
    if (!registered_memory)
    {
      if (require_registered_memory)
      {
        dada_hdu_destroy (hdu);
        hdu = 0;
        throw Error (FailedCall, "dsp::DADABuffer::open_key",
          "cannot register DADA ring buffer blocks as Pinned memory");
      }
      else
        cerr << "WARNING: failed to lock shared memory into RAM and register buffers with the CUDA driver. Check lockable memory limits." << endl;
    }
  }
#endif

  if (passive && dada_hdu_open_view (hdu) < 0)
  {
    dada_hdu_disconnect (hdu);
    dada_hdu_destroy (hdu);
    hdu = 0;
    throw Error (FailedCall, "dsp::DADABuffer::open_key",
    "cannot open DADA ring buffer for viewing");
  }

}

void dsp::DADABuffer::reopen ()
{
  if (verbose)
    cerr << "dsp::DADABuffer::reopen start" << endl;

  assert(hdu != nullptr);
  assert(hdu->data_block != nullptr);

  if (!ipcio_is_open(hdu->data_block))
  {
    if (verbose)
      cerr << "dsp::DADABuffer::reopen call ipcio_open" << endl;

    if (ipcio_open(hdu->data_block, 'R') < 0)
      throw Error(FailedCall, "dsp::DADABuffer::reopen", "ipcio_open failed");
  }

  load_header();

#ifdef HAVE_CUDA
  // check if the input data should be zeroed after reading
  if (ascii_header_get (hdu->header, "ZERO_INPUT", "%u", &zero_input) < 0)
    zero_input = 0;
#endif

  assert(hdu->header != nullptr);

  if (ascii_header_get (hdu->header, "RESOLUTION", "%u", &byte_resolution) < 0)
    byte_resolution = 1;

  // the resolution is the _byte_ resolution; convert to _sample_ resolution
  resolution = get_info()->get_nsamples (byte_resolution);
  if (resolution == 0)
    resolution = 1;

  if (passive && ipcio_view_eod (hdu->data_block, byte_resolution) < 0)
    throw Error (FailedCall, "dsp::DADABuffer::open_key", "cannot ipcio_view_eod");

  if (verbose)
    cerr << "dsp::DADABuffer::reopen rewind" << endl;

  rewind();

  if (verbose)
    cerr << "dsp::DADABuffer::reopen end" << endl;
}

void dsp::DADABuffer::load_header()
{
  if (verbose)
    cerr << "dsp::DADABuffer::load_header calling dada_hdu_open" << endl;

  // dada_hdu_open reads only the header block and leaves data_block unmodified
  if (dada_hdu_open (hdu) < 0)
    throw Error (FailedCall, "dsp::DADABuffer::load_header", "dada_hdu_open failed");

  if (verbose)
    cerr << "dsp::DADABuffer::load_header size=" << hdu->header_size
        << " content=\n" << hdu->header << "\nEND OF HEADER" << endl;

  info = new ASCIIObservation (hdu->header);

  if (!passive && info->get_start_time() == MJD::zero)
  {
    if (verbose)
      cerr << "dsp::DADABuffer::load_header marking header buffer as cleared" << endl;
    ipcbuf_mark_cleared (hdu->header_block);
  }
}

//! Load bytes from shared memory
int64_t dsp::DADABuffer::load_bytes (unsigned char* buffer, uint64_t bytes)
{
  if (verbose)
    cerr << "dsp::DADABuffer::load_bytes ipcio_read " << bytes << " bytes" << endl;

  int64_t bytes_read = ipcio_read (hdu->data_block, (char*)buffer, bytes);
  if (bytes_read < 0)
    cerr << "dsp::DADABuffer::load_bytes error ipcio_read" << endl;

  if (verbose)
    cerr << "dsp::DADABuffer::load_bytes read " << bytes_read << " bytes" << endl;

  return bytes_read;
}

//! Load bytes from shared memory directory to GPU memory
int64_t dsp::DADABuffer::load_bytes_device (unsigned char* device_memory, uint64_t bytes, void * device_handle)
{
#if HAVE_CUDA

  cudaStream_t stream = (cudaStream_t) device_handle;

  if (verbose)
    cerr << "dsp::DADABuffer::load_bytes_device ipcio_read_cuda "
         << bytes << " bytes" << endl;

  if (zero_input)
  {
    if (!zeroed_buffer)
    {
      zeroed_buffer_size = ipcbuf_get_bufsz ((ipcbuf_t*) hdu->data_block);
      if (verbose)
        cerr << "dsp::DADABuffer::load_bytes_device allocating "
             << zeroed_buffer_size << " bytes for zeroing buffer" << endl;
      cudaError_t err = cudaMalloc (&zeroed_buffer, size_t(zeroed_buffer_size));
      if (err != cudaSuccess)
        throw Error (FailedCall, "dsp::DADABuffer::load_bytes_device", "cudaMalloc failed");
      err = cudaMemsetAsync (zeroed_buffer, 0, zeroed_buffer_size, stream);
      if (err != cudaSuccess)
        throw Error (FailedCall, "dsp::DADABuffer::load_bytes_device", "cudaMemsetAsync failed");
    }
  }

  int64_t bytes_read = -1;
#ifdef DADA_IPCIO_READ_ZERO_CUDA  // from psrdada/src/ipcio_cuda.h
  if (zero_input)
    bytes_read = ipcio_read_zero_cuda (hdu->data_block, (char*) device_memory, (char *) zeroed_buffer, bytes, stream);
  else
#endif
    bytes_read = ipcio_read_cuda (hdu->data_block, (char*) device_memory, bytes, stream);
	cudaStreamSynchronize(stream);
  if (bytes_read < 0)
    cerr << "dsp::DADABuffer::load_bytes_device error ipcio_read_cuda" << endl;

  if (verbose)
    cerr << "dsp::DADABuffer::load_bytes_device read " << bytes_read << " bytes" << endl;

  return bytes_read;

#else
  throw Error(InvalidState, "dsp::DADABuffer::load_bytes_device", "not implemented");
#endif
}

//! Adjust the shared memory pointer
int64_t dsp::DADABuffer::seek_bytes (uint64_t bytes)
{
  if (verbose)
    cerr << "dsp::DADABuffer::seek_bytes ipcio_seek "<< bytes << " bytes" << endl;

  int64_t absolute_bytes = ipcio_seek (hdu->data_block, bytes, SEEK_SET);
  if (absolute_bytes < 0)
    cerr << "dsp::DADABuffer::seek_bytes error ipcio_seek" << endl;

  if (verbose)
    cerr << "dsp::DADABuffer::seek_bytes absolute_bytes=" << absolute_bytes << endl;

  return absolute_bytes;
}

void dsp::DADABuffer::seek (int64_t offset, int whence)
{
  if (verbose)
    cerr << "dsp::DADABuffer::seek " << offset << " samples from whence=" << whence << endl;

  if (passive && whence == SEEK_END && offset == 0)
  {
    if (ipcio_view_eod (hdu->data_block, byte_resolution) < 0)
      throw Error (FailedCall, "dsp::DADABuffer::seek", "cannot ipcio_view_eod");
  }
  else
    Input::seek (offset, whence);
}

//! Ensure that block_size is an integer multiple of resolution
void dsp::DADABuffer::set_block_size (uint64_t _size)
{
  if (resolution > 1)
  {
    uint64_t packets = _size / resolution;
    if (_size % resolution)
      packets ++;
    _size = resolution * packets;
  }
  File::set_block_size (_size);
}

//! End-of-data is defined by primary read client (passive viewer)
bool dsp::DADABuffer::eod() const
{
  if (passive)
    return ipcbuf_eod ( &(hdu->data_block->buf) );
  else
    return File::eod();
}

void dsp::DADABuffer::set_total_samples ()
{
}


