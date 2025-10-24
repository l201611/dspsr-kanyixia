
noinst_LTLIBRARIES = libboreas.la

nobase_include_HEADERS = \
  dsp/BoreasSurveyUnpacker.h \
  dsp/BoreasVoltageUnpacker.h

libboreas_la_SOURCES = \
  BoreasSurveyUnpacker.C \
  BoreasVoltageUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += \
  dsp/BoreasVoltageUnpackerCUDA.h

CUDA_SOURCES = BoreasVoltageUnpackerCUDA.cu
libboreas_la_SOURCES += $(CUDA_SOURCES)

endif

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

