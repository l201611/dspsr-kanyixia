
noinst_LTLIBRARIES = libdada.la

nobase_include_HEADERS = dsp/DADABuffer.h \
  dsp/OutputDADABuffer.h

libdada_la_SOURCES = DADABuffer.C \
  OutputDADABuffer.C

libdada_la_LIBADD = @PSRDADA_LIBS@

SUBDIRS = tests

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @PSRDADA_CFLAGS@ @CUDA_CFLAGS@

