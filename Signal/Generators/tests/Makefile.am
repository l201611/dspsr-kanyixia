
check_PROGRAMS = \
  ImpulseTrainTest

ImpulseTrainTest_SOURCES = ImpulseTrainTest.cpp dsp/ImpulseTrainTest.h

include $(top_srcdir)/config/Makefile.local
include $(top_srcdir)/config/Makefile.gtest

AM_CPPFLAGS = -I. $(GTEST_CFLAGS) \
    -I$(top_srcdir)/Signal/Generators \
    -I$(top_srcdir)/Kernel/Classes/testutils \
    -I$(top_builddir)/local_include \
    @PSRCHIVE_CFLAGS@

AM_LDFLAGS = -static

# To avoid dynamically linking unit test executables against previously-installed and outdated shared libraries,
# unit tests should be linked against libtool convenience libraries, which are not installed.

LDADD = \
    $(top_builddir)/Signal/Generators/libGenerators.la \
    $(top_builddir)/Kernel/Classes/libClasses.la \
    $(top_builddir)/Kernel/Classes/testutils/libclasses_testutils.la \
    @PSRCHIVE_LIBS@ \
    $(GTEST_LIBS)

