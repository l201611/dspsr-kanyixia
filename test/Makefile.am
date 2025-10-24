
check_PROGRAMS = test_main

test_main_SOURCES = toml.h catch.hpp test_main.cpp \
                    util/util.cpp util/util.hpp \
                    util/TransformationProxy.cpp util/TransformationProxy.hpp \
                    util/TestConfig.cpp util/TestConfig.hpp \
                    test_Apodization.cpp \
                    test_EventEmitter.cpp \
                    test_FIRFilter.cpp \
                    Convolution/test_Convolution.cpp \
                    Detection/test_Detection.cpp \
                    Filterbank/test_FilterbankEngineCPU.cpp \
                    Filterbank/test_FilterbankConfig.cpp \
                    InverseFilterbank/test_InverseFilterbankEngineCPU.cpp \
                    InverseFilterbank/test_InverseFilterbankConfig.cpp \
                    InverseFilterbank/test_InverseFilterbankResponse.cpp \
                    InverseFilterbank/test_InverseFilterbankPipeline.cpp \
                    InverseFilterbank/InverseFilterbankTestConfig.cpp \
                    PScrunch/test_PScrunch.cpp \
                    SpectralKurtosis/test_SpectralKurtosis.cpp

if HAVE_CUFFT

  test_main_SOURCES += \
        Filterbank/test_FilterbankEngineCUDA.cpp \
        InverseFilterbank/test_InverseFilterbankEngineCUDA_kernels.cpp \
        InverseFilterbank/test_InverseFilterbankEngineCUDA.cpp \
        InverseFilterbank/test_InverseFilterbankEngine_integration.cpp \
        PScrunch/test_PScrunchCUDA.cpp \
        SpectralKurtosis/test_SpectralKurtosis_integration.cpp \
        SpectralKurtosis/SpectralKurtosisTestConfig.hpp \
        util/TestReporter.hpp

  test_main_SOURCES += test_FTransform_cufft_precision.cpp

  check_PROGRAMS += bench_InverseFilterbank
  bench_InverseFilterbank_SOURCES = \
        util/util.cpp \
        util/TestConfig.cpp \
        InverseFilterbank/InverseFilterbankTestConfig.cpp \
        InverseFilterbank/InverseFilterbankTestConfig.hpp \
        InverseFilterbank/bench_InverseFilterbank.cpp

endif

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @PSRCHIVE_CFLAGS@ @CUDA_CFLAGS@

# To avoid dynamically linking unit test executables against previously-installed and outdated shared libraries,
# unit tests should be linked against libtool convenience libraries, which are not installed.

LDADD = $(top_builddir)/Signal/Pulsar/libPulsar.la \
        $(top_builddir)/Signal/General/libGeneral.la \
        $(top_builddir)/Signal/Statistics/libStatistics.la \
	$(top_builddir)/Kernel/Formats/libFormats.la \
        $(top_builddir)/Kernel/Classes/libClasses.la \
        @PSRCHIVE_LIBS@ \
        @CUFFT_LIBS@ @CUDA_LIBS@

if HAVE_CUFFT
if HAVE_CUFFT_CALLBACKS
LDADD += $(top_builddir)/Signal/General/ConvolutionCUDACallbacks_DC.o -lcufft_static -lculibos
else
LDADD += @CUFFT_LIBS@
endif
endif
