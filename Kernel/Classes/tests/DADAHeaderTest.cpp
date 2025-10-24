/***************************************************************************
 *
 *   Copyright (C) 2024 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DADAHeaderTest.h"
#include "dsp/ASCIIObservation.h"
#include "dsp/GtestMain.h"
#include "ascii_header.h"

//! main method passed to googletest
int main(int argc, char *argv[])
{
  return dsp::test::gtest_main(argc, argv);
}

namespace dsp::test
{

  DADAHeaderTest::DADAHeaderTest()
  {
    observation = new ASCIIObservation;

    observation->set_nbit(8);
    observation->set_nchan(32);
    observation->set_npol(2);
    observation->set_ndim(2);
    observation->set_centre_frequency(1024);
    observation->set_bandwidth(64);
  }

  void DADAHeaderTest::SetUp()
  {
  }

  void DADAHeaderTest::TearDown()
  {
  }

  void DADAHeaderTest::assert_header(const char* buffer, unsigned header_size)
  {
    if (header_size == 0)
      header_size = DADAHeader::default_header_size;

    // extract critical meta-data from the DADAHeader
    float hdr_version{0}, hdr_freq{0}, hdr_bw{0}, hdr_tsamp{0};
    uint32_t hdr_size{0}, hdr_nchan{0}, hdr_npol{0}, hdr_ndim{0}, hdr_nbit{0} ;
    ASSERT_EQ(ascii_header_get(buffer, "HDR_SIZE", "%u", &hdr_size), 1);
    ASSERT_EQ(ascii_header_get(buffer, "HDR_VERSION", "%f", &hdr_version), 1);
    ASSERT_EQ(ascii_header_get(buffer, "NCHAN", "%u", &hdr_nchan), 1);
    ASSERT_EQ(ascii_header_get(buffer, "NPOL", "%u", &hdr_npol), 1);
    ASSERT_EQ(ascii_header_get(buffer, "NDIM", "%u", &hdr_ndim), 1);
    ASSERT_EQ(ascii_header_get(buffer, "NBIT", "%u", &hdr_nbit), 1);
    ASSERT_EQ(ascii_header_get(buffer, "FREQ", "%f", &hdr_freq), 1);
    ASSERT_EQ(ascii_header_get(buffer, "BW", "%f", &hdr_bw), 1);

    // assert that the values extracted from the DADAHeader match the input
    ASSERT_EQ(header_size, hdr_size) << " mismatch in HDR_SIZE: input=" << header_size << " output=" << hdr_size;
    ASSERT_EQ(1.0, hdr_version) << " mismatch in HDR_VERSION: input=" << 1.0 << " output=" << hdr_version;
    ASSERT_EQ(observation->get_nchan(), hdr_nchan) << " mismatch in NCHAN: input=" << observation->get_nchan() << " output=" << hdr_nchan;
    ASSERT_EQ(observation->get_npol(), hdr_npol) << " mismatch in NPOL: input=" << observation->get_npol() << " output=" << hdr_npol;
    ASSERT_EQ(observation->get_ndim(), hdr_ndim) << " mismatch in NDIM: input=" << observation->get_ndim() << " output=" << hdr_ndim;
    ASSERT_EQ(observation->get_nbit(), hdr_nbit) << " mismatch in NBIT: input=" << observation->get_nbit() << " output=" << hdr_nbit;
    ASSERT_EQ(observation->get_centre_frequency(), hdr_freq) << " mismatch in FREQ: input=" << observation->get_centre_frequency() << " output=" << hdr_freq;
    ASSERT_EQ(observation->get_bandwidth(), hdr_bw) << " mismatch in BW: input=" << observation->get_bandwidth() << " output=" << hdr_bw;
  }

  TEST_F(DADAHeaderTest, test_construct_delete) // NOLINT
  {
    std::shared_ptr<dsp::DADAHeader> header = std::make_shared<dsp::DADAHeader>();
    ASSERT_NE(header, nullptr);
    header = nullptr;
    ASSERT_EQ(header, nullptr);
  }

  /*
    Verifies that an assertion is thrown if an attempt is made to access the header before it is resized
  */
  TEST_F(DADAHeaderTest, test_get_before_resize) // NOLINT
  {
    std::shared_ptr<dsp::DADAHeader> header = std::make_shared<dsp::DADAHeader>();
    ASSERT_ANY_THROW(header->get_header());
  }

  /*
    Verifies that header sizes are multiples of DADAHeader::default_header_size
  */
  TEST_F(DADAHeaderTest, test_resize) // NOLINT
  {
    std::shared_ptr<dsp::DADAHeader> header = std::make_shared<dsp::DADAHeader>();
    header->resize();
    ASSERT_EQ(header->size(), DADAHeader::default_header_size);
    header->resize(4);
    ASSERT_EQ(header->size(), DADAHeader::default_header_size);
    header->resize(DADAHeader::default_header_size + 4);
    ASSERT_EQ(header->size(), DADAHeader::default_header_size * 2);
  }

  /*
    Verifies that data written by ASCIIObservation can be parsed using ascii_header_get
  */
  TEST_F(DADAHeaderTest, test_ascii_header_get) // NOLINT
  {
    std::shared_ptr<dsp::DADAHeader> header = std::make_shared<dsp::DADAHeader>();
    header->resize();
    observation->unload(header->get_header());
    assert_header(header->get_header());
  }

  /*
    Verifies that data written to header remain intact after resize
  */
  TEST_F(DADAHeaderTest, test_header_after_resize) // NOLINT
  {
    std::shared_ptr<dsp::DADAHeader> header = std::make_shared<dsp::DADAHeader>();
    header->resize();
    observation->unload(header->get_header());
    header->resize(header->size() + 4);
    assert_header(header->get_header(), DADAHeader::default_header_size * 2);
  }

} // namespace dsp::test
