
noinst_LTLIBRARIES = libska1.la

nobase_include_HEADERS = \
  dsp/AAVS2Unpacker.h \
  dsp/CBFPSRUnpacker.h \
  dsp/LFAASPEADUnpacker.h \
  dsp/SKA1Unpacker.h \
  dsp/SKAParallelUnpacker.h

libska1_la_SOURCES = \
  AAVS2Unpacker.C \
  CBFPSRUnpacker.C \
  LFAASPEADUnpacker.C \
  SKA1Unpacker.C \
  SKAParallelUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += \
  dsp/AAVS2UnpackerCUDA.h \
  dsp/LFAASPEADUnpackerCUDA.h \
  dsp/SKA1UnpackerCUDA.h \
  dsp/SKAParallelUnpackerCUDA.h

CUDA_SOURCES = \
  AAVS2UnpackerCUDA.cu \
  LFAASPEADUnpackerCUDA.cu \
  SKA1UnpackerCUDA.cu \
  SKAParallelUnpackerCUDA.cu

libska1_la_SOURCES += $(CUDA_SOURCES)

endif

SUBDIRS = tests

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CUDA_CFLAGS@
