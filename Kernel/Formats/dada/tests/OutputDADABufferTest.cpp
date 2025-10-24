/***************************************************************************
 *
 *   Copyright (C) 2024-2025 by Jesmigel Cantos and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Detection.h"
#include "dsp/GtestMain.h"
#include "dsp/OutputDADABufferTest.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <ipcbuf.h>
#include <ascii_header.h>

//! main method passed to googletest
int main(int argc, char* argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

using namespace std;

namespace dsp::test {

OutputDADABufferTest::OutputDADABufferTest()
{
  // defaults to 4096
  template_header.resize();

  if (ascii_header_set(template_header.get_header(), "TEST_FLOAT", "%f", test_float_value) < 0)
    throw Error (InvalidState, "dsp::test::OutputDADABufferTest ctor", "failed to set TEST_FLOAT");
}

void OutputDADABufferTest::SetUp()
{
  // dsp::Operation::verbose = true;
  // dsp::Observation::verbose = true;
  init_input();
}

void OutputDADABufferTest::TearDown()
{
  clear_input();
}

auto OutputDADABufferTest::convert_key(const std::string &key_string) -> key_t
{
  std::stringstream ss;
  key_t key = 0;
  ss << std::hex << key_string;
  ss >> key >> std::dec;
  return key;
}

void OutputDADABufferTest::setup_smrb(const std::string &smrb_key)
{
  key_t data_block_key = convert_key(smrb_key);
  key_t header_block_key = data_block_key+1;

  int device_id=-1;

  // create data block
  if (ipcbuf_create_work(reinterpret_cast<ipcbuf_t *>(&data_block), data_block_key, data_nbufs, data_bufsz, num_readers, device_id) != 0)
  {
    cerr << "dsp::OutputDADABufferTest::setup_smrb Unable to create data block" << endl;
  }

  // create header block
  if (ipcbuf_create(&header_block, header_block_key, hdr_nbufs, hdr_bufsz, num_readers) != 0)
  {
    cerr << "dsp::OutputDADABufferTest::setup_smrb Unable to create header block" << endl;
  }
}

void OutputDADABufferTest::connect(std::string data_key)
{
  key_t smrb_key = convert_key(data_key);
  if (dsp::Operation::verbose)
    cerr << "dsp::OutputDADABufferTest::connect dada_hdu_set_key(hdu, " << smrb_key << ")" << endl;
  dada_hdu_set_key(hdu, smrb_key);
  if (!hdu_connected)
  {
    if (dsp::Operation::verbose)
      cerr << "dsp::OutputDADABufferTest::connect dada_hdu_connect()" << endl;
    if (dada_hdu_connect(hdu) < 0)
      throw Error (InvalidState, "dsp::OutputDADABuffer::connect", "cannot connect to DADA ring buffers");
    hdu_connected = true;
    if (dsp::Operation::verbose)
      cerr << "dsp::OutputDADABufferTest::connect hdu_connected=true" << endl;

    if (!hdu_locked)
    {
      if (dsp::Operation::verbose)
        cerr << "dsp::OutputDADABufferTest::connect dada_hdu_lock_read(hdu)" << endl;
      if (dada_hdu_lock_read(hdu) != 0)
        throw Error (InvalidState, "dsp::OutputDADABuffer::connect", "cannot lock read to DADA ring buffers");
      hdu_locked = true;
      if (dsp::Operation::verbose)
        cerr << "dsp::OutputDADABufferTest::connect hdu_locked=true" << endl;
    }
  }
}

void OutputDADABufferTest::disconnect()
{
  if (hdu_connected)
  {
    if (hdu_locked)
    {
      if (dada_hdu_unlock_read (hdu) < 0)
        cerr << "dsp::OutputDADABuffer::disconnect dada_hdu_unlock_write failed" << endl;
      hdu_locked=false;
    }
    if (dada_hdu_disconnect (hdu) < 0)
      cerr << "dsp::OutputDADABuffer::disconnect dada_hdu_disconnect failed" << std::endl;
    hdu_connected=false;
  }
}

void OutputDADABufferTest::teardown_smrb()
{
  disconnect();
  dada_hdu_destroy(hdu);
  hdu_opened = false;
  ipcbuf_destroy(&header_block);
  ipcbuf_destroy(reinterpret_cast<ipcbuf_t *>(&data_block));
}

void OutputDADABufferTest::fill_input()
{
  input->set_swap(false);
  input->set_nsub_swap(0);
  input->set_nchan(nchan);
  input->set_npol(npol);
  input->set_ndim(ndim);
  input->set_nbit(nbit);

  input->resize(ndat);
  uint8_t *inptr = input->get_rawptr();

  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
    {
      uint64_t offset = idat * nchan * npol * ndim + ichan * npol * ndim;
      // unsigned char *inptr = input->get_rawptr() + offset;
      for (unsigned ipol=0; ipol<input->get_npol(); ipol++)
      {
        for (unsigned idim=0; idim<input->get_ndim(); idim++)
        {
          if (offset % 2 != 0)
          {
            inptr[offset] = (uint8_t)ipol;
          }
          else
          {
            inptr[offset] = (uint8_t)ichan;
          }
          // cerr << "inptr["<<offset<<"]="<<(unsigned int)inptr[offset]<<std::endl;
        } // dim
      }   // pol
    }     // chan
  }       // time
}

void OutputDADABufferTest::assert_header()
{
  uint64_t bytes{};
  char * buf = ipcbuf_get_next_read(&header_block, &bytes);
  if (dsp::Operation::verbose)
    cerr << "dsp::OutputDADABufferTest::assert_header buf=" << reinterpret_cast<void*>(buf)
         << "bytes=" << bytes << " hdu->header_block_key=" << hdu->header_block_key << endl;
  ASSERT_TRUE(hdu->header_size==0);
  if (!hdu_opened)
  {
    if (dsp::Operation::verbose)
      cerr << "dsp::OutputDADABufferTest::assert_header opening hdu" << endl;
    if (dada_hdu_open(hdu)!=0)
      throw Error (InvalidState, "dsp::OutputDADABuffer::connect", "cannot lock read to DADA ring buffers");
    hdu_opened = true;
    if (dsp::Operation::verbose)
      cerr << "dsp::OutputDADABufferTest::assert_header hdu opened" << endl;
  }

  if (dsp::Operation::verbose)
    cerr << "dsp::OutputDADABufferTest::assert_header bytes=" << bytes << " header_size=" << hdu->header_size << endl;
  ASSERT_TRUE(hdu->header_size > 0);
  ASSERT_TRUE(bytes > 0);

  unsigned smrb_nchan{0}, smrb_npol{0}, smrb_nbit, smrb_ndim{0}, smrb_resolution{0};
  float smrb_test_float{0};
  ASSERT_TRUE(0 < ascii_header_get(hdu->header, "NCHAN", "%d", &smrb_nchan));
  ASSERT_TRUE(0 < ascii_header_get(hdu->header, "NPOL", "%d", &smrb_npol));
  ASSERT_TRUE(0 < ascii_header_get(hdu->header, "NBIT", "%d", &smrb_nbit));
  ASSERT_TRUE(0 < ascii_header_get(hdu->header, "NDIM", "%d", &smrb_ndim));
  ASSERT_TRUE(0 < ascii_header_get(hdu->header, "RESOLUTION", "%d", &smrb_resolution));
  ASSERT_TRUE(0 < ascii_header_get(hdu->header, "TEST_FLOAT", "%f", &smrb_test_float));

  ASSERT_EQ(input->get_nchan(), smrb_nchan);
  ASSERT_EQ(input->get_npol(), smrb_npol);
  ASSERT_EQ(input->get_nbit(), smrb_nbit);
  ASSERT_EQ(input->get_ndim(), smrb_ndim);
  ASSERT_EQ(smrb_nchan * smrb_npol * smrb_ndim *  smrb_nbit / 8, smrb_resolution);
  ASSERT_EQ(test_float_value, smrb_test_float);
}

void OutputDADABufferTest::assert_data()
{
  int64_t bufsz = ipcbuf_get_bufsz(&hdu->data_block->buf);
  if (dsp::Operation::verbose)
    cerr << "dsp::OutputDADABufferTest::assert_data bufsz=" << bufsz << endl;
  ASSERT_TRUE(bufsz > 0);

  uint64_t bytes = nchan * npol * ndim * nbit * ndat / 8;
  if (dsp::Operation::verbose)
    cerr << "dsp::OutputDADABufferTest::assert_data bytes=" << bytes << endl;
  std::vector<uint8_t> data_array (bytes, 0);
  uint8_t* buffer = &data_array[0];
  if (!hdu_opened)
  {
    if (dsp::Operation::verbose)
      cerr << "dsp::OutputDADABufferTest::assert_data opening hdu" << endl;
    if (dada_hdu_open(hdu)!=0)
      throw Error (InvalidState, "dsp::OutputDADABuffer::connect", "cannot lock read to DADA ring buffers");
    hdu_opened = true;
    if (dsp::Operation::verbose)
      cerr << "dsp::OutputDADABufferTest::assert_data hdu opened" << endl;
  }
  if (dsp::Operation::verbose)
    cerr << "dsp::OutputDADABufferTest::assert_data ipcio_read()" << endl;
  int64_t bytes_read = ipcio_read (hdu->data_block, (char*)buffer, bytes);
  if (dsp::Operation::verbose)
    cerr << "dsp::OutputDADABufferTest::assert_data validate_data bytes_read=" << bytes_read << endl;

  const unsigned char *inptr = input->get_rawptr();
  // strong assertion for data ordering
  for (uint64_t idat=0; idat<ndat; idat++)
  {
    for (unsigned ichan=0; ichan<nchan; ichan++)
    {
      uint64_t offset = idat * nchan * npol * ndim + ichan * npol * ndim;
      for (unsigned ipol=0; ipol<npol; ipol++)
      {
        for (unsigned idim=0; idim<ndim; idim++)
        {
          ASSERT_EQ(inptr[offset], buffer[offset]);
        }
      }
    }
  }
}

void OutputDADABufferTest::init_input()
{
  // INITIALISE hdu
  multilog_t* log = multilog_open ("OutputDADABufferTest", 0);
  multilog_add (log, stderr);
  // create the DADA HDU structure and initalise it
  hdu = dada_hdu_create (log);

  // INITIALISE SMRB
  setup_smrb(smrb_key);
  input = new dsp::BitSeries;

  fill_input();
}

void OutputDADABufferTest::clear_input()
{
  odb = nullptr;
  input = nullptr;
  teardown_smrb();
}

TEST_F(OutputDADABufferTest, test_create_destroy) // NOLINT
{
  clear_input();
  ASSERT_ANY_THROW(odb = std::make_shared<OutputDADABuffer>(smrb_key));
  setup_smrb(smrb_key);
  ASSERT_NO_THROW(odb = std::make_shared<OutputDADABuffer>(smrb_key)); // segfaults without setup_smrb
  ASSERT_ANY_THROW(odb->set_input(input));
  ASSERT_ANY_THROW(odb->calculation());
  input = new dsp::BitSeries;
  fill_input();
  ASSERT_NO_THROW(odb->set_input(input));
  ASSERT_NO_THROW(odb->calculation());
}

TEST_F(OutputDADABufferTest, test_single_large_block) // NOLINT
{
  odb = std::make_shared<OutputDADABuffer>(smrb_key);

  // copy the "template" header
  odb->get_header()->set_header(template_header.get_header());
  odb->set_input(input);
  odb->calculation();
  connect(smrb_key);
  assert_header();
  odb = nullptr;
  assert_data();
}

TEST_F(OutputDADABufferTest, test_multiple_blocks)
{
  clear_input();
  // RECREATE SMRB
  data_nbufs = 2;
  ndat *= data_nbufs;
  init_input();
  odb = std::make_shared<OutputDADABuffer>(smrb_key);

  // copy the "template" header
  odb->get_header()->set_header(template_header.get_header());
  odb->set_input(input);
  odb->calculation();
  connect(smrb_key);
  assert_header();
  odb = nullptr;
  assert_data();
}

} // namespace dsp::test
