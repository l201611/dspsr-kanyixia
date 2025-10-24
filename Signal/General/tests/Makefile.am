
check_PROGRAMS = \
  ChanPolSelectTest \
  ConvolutionTest \
  InverseFilterbankTest \
  LoadToQuantizeTest \
  RescaleTest \
  RescaleMedianMadCalculatorTest \
  ResponseTest \
  ResponseProductTest \
  TScrunchTest

ChanPolSelectTest_SOURCES = ChanPolSelectTest.cpp dsp/ChanPolSelectTest.h
ConvolutionTest_SOURCES = ConvolutionTest.cpp dsp/ConvolutionTest.h
InverseFilterbankTest_SOURCES = InverseFilterbankTest.cpp dsp/InverseFilterbankTest.h
LoadToQuantizeTest_SOURCES = LoadToQuantizeTest.cpp dsp/LoadToQuantizeTest.h
RescaleTest_SOURCES = RescaleTest.cpp dsp/RescaleTest.h
RescaleMedianMadCalculatorTest_SOURCES = RescaleMedianMadCalculatorTest.cpp dsp/RescaleMedianMadCalculatorTest.h
TScrunchTest_SOURCES = TScrunchTest.cpp dsp/TScrunchTest.h
ResponseTest_SOURCES = ResponseTest.cpp dsp/ResponseTest.h
ResponseProductTest_SOURCES = ResponseProductTest.cpp dsp/ResponseProductTest.h

include $(top_srcdir)/config/Makefile.local
include $(top_srcdir)/config/Makefile.gtest

AM_CPPFLAGS = -I. $(GTEST_CFLAGS) \
    -I$(top_srcdir)/Signal/General/testutils \
    -I$(top_srcdir)/Kernel/Classes/testutils \
    -I$(top_builddir)/local_include \
    @PSRCHIVE_CFLAGS@

AM_LDFLAGS = -static

# To avoid dynamically linking unit test executables against previously-installed and outdated shared libraries,
# unit tests should be linked against libtool convenience libraries, which are not installed.

LDADD = \
    $(top_builddir)/Signal/General/libGeneral.la \
    $(top_builddir)/Signal/Statistics/libStatistics.la \
    $(top_builddir)/Signal/General/testutils/libsignal_testutils.la \
    $(top_builddir)/Kernel/Classes/testutils/libclasses_testutils.la \
    $(top_builddir)/Kernel/libdspbase_local.la \
    $(GTEST_LIBS) \
    @PSRCHIVE_LIBS@ \
    @CUDA_LIBS@ @CUFFT_LIBS@

