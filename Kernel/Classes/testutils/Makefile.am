
check_LTLIBRARIES = libclasses_testutils.la

libclasses_testutils_la_SOURCES = \
  GtestMain.cpp dsp/GtestMain.h \
  TestSource.cpp dsp/TestSource.h \
  GaussianNoiseSource.cpp dsp/GaussianNoiseSource.h \
  ImpulsiveNoiseSource.cpp dsp/ImpulsiveNoiseSource.h \
  KnownDataSource.cpp dsp/KnownDataSource.h \
  SumSource.cpp dsp/SumSource.h

if HAVE_CUDA
libclasses_testutils_la_SOURCES += \
  TransferCUDATestHelper.cpp dsp/TransferCUDATestHelper.h
endif

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.gtest

# To avoid dynamically linking unit test executables against previously-installed and outdated shared libraries,
# unit tests should be linked against libtool convenience libraries, which are not installed.

AM_CPPFLAGS += -I. $(GTEST_CFLAGS) \
  -I$(top_srcdir)/Kernel/Classes

