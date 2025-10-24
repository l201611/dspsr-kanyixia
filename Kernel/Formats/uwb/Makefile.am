
noinst_LTLIBRARIES = libuwb.la

nobase_include_HEADERS =  dsp/UWBUnpacker.h dsp/UWBFloatUnpacker.h dsp/UWBFourBitUnpacker.h dsp/UWBEightBitUnpacker.h dsp/UWBTwoBitUnpacker.h

libuwb_la_SOURCES = UWBUnpacker.C UWBFloatUnpacker.C UWBFourBitUnpacker.C UWBEightBitUnpacker.C UWBTwoBitUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/UWBUnpackerCUDA.h dsp/UWBFloatUnpackerCUDA.h

CUDA_SOURCES = UWBUnpackerCUDA.cu UWBFloatUnpackerCUDA.cu
libuwb_la_SOURCES += $(CUDA_SOURCES)

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CUDA_CFLAGS@
