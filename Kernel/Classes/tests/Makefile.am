
check_PROGRAMS = \
  DADAHeaderTest \
  DADAOutputFileTest \
  GenericVoltageDigitizerTest \
  GaussianNoiseSourceTest \
  ImpulsiveNoiseSourceTest \
  SumSourceTest \
  WeightedTimeSeriesTest

DADAHeaderTest_SOURCES = DADAHeaderTest.cpp dsp/DADAHeaderTest.h
DADAOutputFileTest_SOURCES = DADAOutputFileTest.cpp dsp/DADAOutputFileTest.h
GenericVoltageDigitizerTest_SOURCES = GenericVoltageDigitizerTest.cpp dsp/GenericVoltageDigitizerTest.h
GaussianNoiseSourceTest_SOURCES = GaussianNoiseSourceTest.cpp dsp/GaussianNoiseSourceTest.h
ImpulsiveNoiseSourceTest_SOURCES = ImpulsiveNoiseSourceTest.cpp dsp/ImpulsiveNoiseSourceTest.h
SumSourceTest_SOURCES = SumSourceTest.cpp dsp/SumSourceTest.h
WeightedTimeSeriesTest_SOURCES = WeightedTimeSeriesTest.cpp dsp/WeightedTimeSeriesTest.h

include $(top_srcdir)/config/Makefile.local
include $(top_srcdir)/config/Makefile.gtest

AM_CPPFLAGS = -I. $(GTEST_CFLAGS) \
    -I$(top_srcdir)/Kernel/Classes/testutils \
    -I$(top_srcdir)/Kernel/Classes \
    -I$(top_builddir)/local_include \
    @PSRCHIVE_CFLAGS@

AM_LDFLAGS = -static

# To avoid dynamically linking unit test executables against previously-installed and outdated shared libraries,
# unit tests should be linked against libtool convenience libraries, which are not installed.

LDADD = \
    $(top_builddir)/Kernel/Classes/testutils/libclasses_testutils.la \
    $(top_builddir)/Kernel/Classes/libClasses.la \
    @PSRCHIVE_LIBS@ \
    @CUDA_LIBS@ \
    @CUFFT_LIBS@ \
    $(GTEST_LIBS)

