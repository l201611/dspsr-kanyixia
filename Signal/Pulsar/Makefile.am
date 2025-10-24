
SUBDIRS = . tests

noinst_LTLIBRARIES = libPulsar.la

nobase_include_HEADERS = \
dsp/Archiver.h                  dsp/Subint.h \
dsp/Fold.h			dsp/FoldManager.h \
dsp/TimeDivide.h		dsp/UnloaderShare.h \
dsp/LoadToFold1.h               dsp/PhaseLockedFilterbank.h \
dsp/LoadToFoldConfig.h          dsp/PhaseSeries.h \
dsp/LoadToFoldN.h               dsp/PhaseSeriesUnloader.h \
dsp/CyclicFold.h

libPulsar_la_SOURCES = \
Archiver.C                            \
ArchiverExtensions.C    TimeDivide.C            \
Fold.C                  FoldManager.C \
UnloaderShare.C \
LoadToFold1.C           PhaseLockedFilterbank.C \
LoadToFoldConfig.C      PhaseSeries.C  \
LoadToFoldN.C           PhaseSeriesUnloader.C \
CyclicFold.C

if HAVE_CUFFT

nobase_include_HEADERS += dsp/FoldCUDA.h dsp/TransferPhaseSeriesCUDA.h \
			  dsp/CyclicFoldEngineCUDA.h

CUDA_SOURCES = FoldCUDA.cu CyclicFoldEngineCUDA.cu
libPulsar_la_SOURCES += TransferPhaseSeriesCUDA.C $(CUDA_SOURCES)

endif

bin_PROGRAMS = dspsr

if HAVE_TEMPO2
bin_PROGRAMS += t2pred
t2pred_SOURCES = t2pred.C
endif

dspsr_SOURCES = dspsr.C 

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

LDADD = libPulsar.la \
	$(top_builddir)/Signal/General/libGeneral.la \
	$(top_builddir)/Kernel/libdspbase.la \
	$(top_builddir)/Signal/Statistics/libStatistics.la \
	@CUDA_LIBS@ 

if HAVE_CUFFT_CALLBACKS
LDADD += $(top_builddir)/Signal/General/ConvolutionCUDACallbacks_DC.o -lcufft_static -lculibos
else
LDADD += @CUFFT_LIBS@
endif

AM_CPPFLAGS += @CUFFT_CFLAGS@
if HAVE_CFITSIO
AM_CPPFLAGS += @CFITSIO_CFLAGS@
endif
AM_CXXFLAGS = @OPENMP_CFLAGS@

