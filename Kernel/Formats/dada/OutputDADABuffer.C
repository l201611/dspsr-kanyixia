//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel A. Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ASCIIObservation.h"
#include "dsp/OutputDADABuffer.h"

// psrdada
#include "dada_def.h"
#include "ipcio.h"
#include "ipcbuf.h"

#include <vector>
#include <cstring>
#include <cassert>

dsp::OutputDADABuffer::OutputDADABuffer (const std::string &key_string)
  : Sink<BitSeries> ("OutputDADABuffer")
{
  multilog_t* log = multilog_open ("dspsr-output-buffer", 0);
  multilog_add (log, stderr);

  // create the DADA HDU structure and initialise it
  hdu = dada_hdu_create (log);
  key_t smrb_key = parse_psrdada_key(key_string);
  connect(smrb_key);
}

dsp::OutputDADABuffer::~OutputDADABuffer ()
{
  disconnect();
}

auto dsp::OutputDADABuffer::parse_psrdada_key(const std::string &key_string) -> key_t
{
  std::stringstream ss;
  key_t key = 0;
  ss << std::hex << key_string;
  ss >> key >> std::dec;
  return key;
}

void dsp::OutputDADABuffer::connect(key_t smrb_key)
{
#ifdef _DEBUG
  std::cerr << "setting key [" << smrb_key << "]" << std::endl;
#endif
  dada_hdu_set_key(hdu, smrb_key);

#ifdef _DEBUG
  std::cerr << "connect [" << smrb_key << "]" << std::endl;
#endif
  if (dada_hdu_connect (hdu) < 0) // this segfaults when smrb doesn't exist
    throw Error (InvalidState, "dsp::OutputDADABuffer::connect",
		 "cannot connect to DADA ring buffers");
  connected = true;

#ifdef _DEBUG
  std::cerr << "connected [" << smrb_key << "]" << std::endl;
  std::cerr << "lock [" << smrb_key << "]" << std::endl;
#endif

  if (dada_hdu_lock_write (hdu) != 0)
    throw Error (InvalidState, "dsp::OutputDADABuffer::connect",
		 "cannot lock write to DADA ring buffers");

  locked=true;

#ifdef _DEBUG
  std::cerr << "locked [" << smrb_key << "]" << std::endl;
#endif
}

void dsp::OutputDADABuffer::reopen()
{
  if (verbose)
    std::cerr << "dsp::OutputDADABuffer::reopen" << std::endl;

  assert(hdu != nullptr);

  if (!ipcio_is_open(hdu->data_block))
  {
    if (verbose)
      std::cerr << "dsp::OutputDADABuffer::reopen call ipcio_open" << std::endl;

    if (ipcio_open(hdu->data_block, 'W') < 0)
      throw Error(FailedCall, "dsp::OutputDADABuffer::reopen", "ipcio_open failed");
  }

  header_written = false;
}

void dsp::OutputDADABuffer::close()
{
  if (verbose)
    std::cerr << "dsp::OutputDADABuffer::close" << std::endl;

  assert(hdu != nullptr);

  if (ipcio_is_open(hdu->data_block))
  {
    if (verbose)
      std::cerr << "dsp::OutputDADABuffer::close call ipcio_close" << std::endl;

    if (ipcio_close(hdu->data_block) < 0)
    {
      throw Error (FailedCall, "dsp::OutputDADABuffer::close", "failed call to ipcio_close");
    }
  }

#if DADA_MAJOR_VERSION >= 1 && DADA_MINOR_VERSION >=1
  /*
    After closing a transfer, re-lock write access to the header block to ensure
    that the state of the header is WCHANGE such that, should another transfer
    be written to the header, it will advance to the next transfer.
  */
  if (ipcbuf_relock_write(hdu->header_block) > 0)
  {
    throw Error (FailedCall, "dsp::OutputDADABuffer::close", "failed call to ipcbuf_relock_write on the header_block");
  }
#endif
}

void dsp::OutputDADABuffer::disconnect()
{
  if (verbose)
    std::cerr << "dsp::OutputDADABuffer::disconnect" << std::endl;

  assert(hdu != nullptr);

  if (locked && dada_hdu_unlock_write (hdu) < 0)
    std::cerr << "dsp::OutputDADABuffer::disconnect dada_hdu_unlock_write failed" << std::endl;

  locked = false;

  if (connected && dada_hdu_disconnect (hdu) < 0)
    std::cerr << "dsp::OutputDADABuffer::disconnect dada_hdu_disconnect failed" << std::endl;

  connected = false;
}

void dsp::OutputDADABuffer::write_header()
{
  if (verbose)
    std::cerr << "dsp::OutputDADABuffer::write_header" << std::endl;

  size_t header_size = ipcbuf_get_bufsz (hdu->header_block);
  char * header = ipcbuf_get_next_write (hdu->header_block);

  // in binary mode, write a 4k ASCII (DADA) header
  dsp::ASCIIObservation ascii (input);

  if (dada_header.size() == 0)
    dada_header.resize();

  char* buffer = dada_header.get_header();

  ascii.unload(buffer);

  if (ascii_header_set (buffer, "HDR_SIZE", "%d", header_size) != 0)
    throw Error (InvalidState, "dsp::OutputDADABuffer::write_header",
    "failed to set HDR_SIZE in output file header");

  static constexpr unsigned bits_per_byte = 8;
  unsigned bytes_per_sample = ascii.get_nchan() * ascii.get_npol() * ascii.get_ndim() * ascii.get_nbit() / bits_per_byte;
  if (dsp::Operation::verbose)
    std::cerr << "dsp::OutputDADABuffer::write_header nchan=" << ascii.get_nchan() << " npol=" << ascii.get_npol()
              << " ndim=" << ascii.get_ndim() << " nbit=" << ascii.get_nbit()
              << " RESOLUTION=" << bytes_per_sample << std::endl;

  // set the RESOLUTION header parameter, under the assumption of TFP ordering
  if (ascii_header_set (buffer, "RESOLUTION", "%u", bytes_per_sample) != 0)
    throw Error (InvalidState, "dsp::OutputDADABuffer::write_header",
    "failed to set RESOLUTION=%u in output file header", bytes_per_sample);

  // ensure the header is initialized with the null character string
  memset(header, '\0', header_size);

  assert(dada_header.size() <= header_size);
  memcpy(header, buffer, dada_header.size());

#ifdef _DEBUG
  std::cerr << "write_header header_size: " << header_size << std::endl;
  std::cerr << "write_header hdu->header_block_key: " << hdu->header_block_key << std::endl;
  std::cerr << "write_header hdu->data_block_key: " << hdu->data_block_key << std::endl;
#endif

#if DADA_MAJOR_VERSION >= 1 && DADA_MINOR_VERSION >=1
  /*
    Before marking the header as filled, set the EOD flag to true,
    indicating that the meta-data for this transfer is complete.
  */
  if (ipcbuf_enable_eod (hdu->header_block) != 0) {
    throw Error (InvalidState, "dsp::OutputDADABuffer::write_header",
    "failed to mark end of data on the header block.");
  }
#endif

  if (ipcbuf_mark_filled (hdu->header_block, header_size) != 0)  {
    throw Error (InvalidState, "dsp::OutputDADABuffer::write_header",
    "failed to mark the header as filled.");
  }
}

void dsp::OutputDADABuffer::calculation()
{
  if (!header_written)
    write_header();

  header_written = true;

  // call unload bytes on the whole bit series
  unload_bytes(input->get_rawptr(), input->get_nbytes());
}

void dsp::OutputDADABuffer::unload_bytes(const void* buffer, uint64_t nbytes)
{
  // write to the ring buffer using the ipcio library
  int bytes_written = ipcio_write(hdu->data_block, (char*)buffer, nbytes);

#ifdef _DEBUG
  std::cerr << "unload_bytes nbytes: " << nbytes << std::endl;
  std::cerr << "unload_bytes hdu->data_block_key: " << hdu->data_block_key << std::endl;
  std::cerr << "unload_bytes bytes_written: " << bytes_written << std::endl;
#endif

  if (bytes_written < 0)
    throw Error (FailedCall, "dsp::OutputDADABuffer::unload_bytes",
      "failed to write to data block through ipcio_write");
}

void dsp::OutputDADABuffer::display_input_contents()
{
  unsigned nchan = input->get_nchan();
  unsigned npol = input->get_npol();
  unsigned ndim = input->get_ndim();
  unsigned nbit = input->get_nbit();
  uint64_t ndat = input->get_ndat();
  std::cerr << "nchan: " << nchan << std::endl;
  std::cerr << "npol: " << npol << std::endl;
  std::cerr << "ndim: " << ndim << std::endl;
  std::cerr << "nbit: " << nbit << std::endl;
  std::cerr << "ndat: " << ndat << std::endl;

  const unsigned char *inptr = input->get_rawptr();
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      uint64_t offset = idat * nchan * npol * ndim + ichan * npol * ndim;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned idim=0; idim<ndim; idim++)
        {
          std::cerr << "input[" << offset << "]: " << static_cast<unsigned int>(inptr[offset]) << std::endl;
        }
      }
    }
  }
}
