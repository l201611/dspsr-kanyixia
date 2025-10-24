
check_PROGRAMS = OutputDADABufferTest

OutputDADABufferTest_SOURCES = OutputDADABufferTest.cpp dsp/OutputDADABufferTest.h

include $(top_srcdir)/config/Makefile.local
include $(top_srcdir)/config/Makefile.gtest

AM_CPPFLAGS = -I. $(GTEST_CFLAGS) \
    -I$(top_srcdir)/Kernel/Classes/testutils \
    -I$(top_builddir)/local_include \
    @PSRDADA_CFLAGS@ \
    @PSRCHIVE_CFLAGS@

AM_LDFLAGS = -static

# To avoid dynamically linking unit test executables against previously-installed and outdated shared libraries,
# unit tests should be linked against libtool convenience libraries, which are not installed.

LDADD = \
    $(top_builddir)/Kernel/Formats/dada/libdada.la \
    $(top_builddir)/Kernel/Classes/testutils/libclasses_testutils.la \
    $(top_builddir)/Kernel/Classes/libClasses.la \
    $(GTEST_LIBS) \
    @PSRDADA_LIBS@ \
    @PSRCHIVE_LIBS@

