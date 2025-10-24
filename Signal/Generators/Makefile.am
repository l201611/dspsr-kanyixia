
check_LTLIBRARIES = libGenerators.la

nobase_include_HEADERS = \
  dsp/Generator.h \
  dsp/ImpulseTrain.h \
  dsp/Superposition.h

libGenerators_la_SOURCES = \
  Generator.C \
  ImpulseTrain.C \
  Superposition.C

SUBDIRS = . tests

include $(top_srcdir)/config/Makefile.include

LDADD = libGenerators.la \
        $(top_builddir)/Kernel/libdspbase.la
