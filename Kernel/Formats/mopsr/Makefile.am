
noinst_LTLIBRARIES = libmopsr.la

nobase_include_HEADERS =  dsp/MOPSRUnpacker.h

libmopsr_la_SOURCES = MOPSRUnpacker.C

include $(top_srcdir)/config/Makefile.include

if HAVE_CUDA

nobase_include_HEADERS += dsp/MOPSRUnpackerCUDA.h

CUDA_SOURCES = MOPSRUnpackerCUDA.cu
libmopsr_la_SOURCES += $(CUDA_SOURCES)

include $(top_srcdir)/config/Makefile.cuda
endif


AM_CPPFLAGS += @CUDA_CFLAGS@
