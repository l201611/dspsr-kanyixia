
check_LTLIBRARIES = libsignal_testutils.la

libsignal_testutils_la_SOURCES = \
  RescaleScaleOffsetDumpTestHelper.cpp dsp/RescaleScaleOffsetDumpTestHelper.h \
  ResponseTestHelper.cpp dsp/ResponseTestHelper.h \
  dsp/SignalStateTestHelper.h \
  TestResponse.cpp dsp/TestResponse.h \
  dsp/TransformationTestHelper.h

include $(top_srcdir)/config/Makefile.local
include $(top_srcdir)/config/Makefile.gtest

AM_CPPFLAGS = -I. $(GTEST_CFLAGS) \
    -I$(top_srcdir)/Kernel/Classes/testutils \
    -I$(top_builddir)/local_include \
    @PSRCHIVE_CFLAGS@

AM_LDFLAGS = -static

# To avoid dynamically linking unit test executables against previously-installed and outdated shared libraries,
# unit tests should be linked against libtool convenience libraries, which are not installed.

LDADD = \
    $(top_builddir)/Kernel/Classes/testutils/libclasses_testutils.la \
    $(top_builddir)/Signal/General/libGeneral.la \
    $(top_builddir)/Signal/Statistics/libStatistics.la \
    $(top_builddir)/Kernel/Classes/libClasses.la \
    $(GTEST_LIBS) \
    @PSRCHIVE_LIBS@ \
    @CUDA_LIBS@ @CUFFT_LIBS@
