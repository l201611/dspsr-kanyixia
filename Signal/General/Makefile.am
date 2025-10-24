
noinst_LTLIBRARIES = libGeneral.la

nobase_include_HEADERS = \
  dsp/ACFilterbank.h \
  dsp/Accumulator.h \
  dsp/Apodization.h \
  dsp/AutoCorrelation.h \
  dsp/Bandpass.h \
  dsp/BandpassMonitor.h \
  dsp/BitStatsPlotter.h \
  dsp/Buffer.h \
  dsp/CanSliceTime.h \
  dsp/ChanPolSelect.h \
  dsp/Chomper.h \
  dsp/Convolution.h \
  dsp/Derotation.h \
  dsp/Dedispersion.h \
  dsp/DedispersionHistory.h \
  dsp/DedispersionPipe.h \
  dsp/DedispersionPipeConfig.h \
  dsp/DedispersionSampleDelay.h \
  dsp/DelayStartTime.h \
  dsp/Detection.h \
  dsp/Dump.h \
  dsp/DynamicFilter.h \
  dsp/Example.h \
  dsp/ExamplePipeline.h \
  dsp/ExcisionHistoryPlotter.h \
  dsp/ExcisionStatsPlotter.h \
  dsp/FScrunch.h \
  dsp/FZoom.h \
  dsp/Filterbank.h \
  dsp/FilterbankBench.h \
  dsp/FilterbankConfig.h \
  dsp/FilterbankEngine.h \
  dsp/FilterbankEngineCPU.h \
  dsp/FourthMoment.h \
  dsp/GeometricDelay.h \
  dsp/IncoherentFilterbank.h \
  dsp/InverseFilterbank.h \
  dsp/InverseFilterbankConfig.h \
  dsp/InverseFilterbankEngine.h \
  dsp/InverseFilterbankEngineCPU.h \
  dsp/InverseFilterbankResponse.h \
  dsp/LevelHistory.h \
  dsp/LevelMonitor.h \
  dsp/LoadToSlice.h \
  dsp/LoadToSliceN.h \
  dsp/LoadToQuantize.h \
  dsp/Mask.h \
  dsp/MaskTimes.h \
  dsp/MultiThread.h \
  dsp/OptimalFFT.h \
  dsp/OptimalFilterbank.h \
  dsp/PScrunch.h \
  dsp/Pipeline.h \
  dsp/PipelinePerformanceMetrics.h \
  dsp/PlasmaResponse.h \
  dsp/PolnCalibration.h \
  dsp/PolnReshape.h \
  dsp/PolnSelect.h \
  dsp/RFIFilter.h \
  dsp/RFIZapper.h \
  dsp/Rescale.h \
  dsp/RescaleScaleOffsetDump.h \
  dsp/RescaleMeanStdCalculator.h \
  dsp/RescaleMedianMadCalculator.h \
  dsp/Resize.h \
  dsp/Response.h \
  dsp/ResponseProduct.h \
  dsp/SKComputer.h \
  dsp/SKDetector.h \
  dsp/SKFilterbank.h \
  dsp/SKMasker.h \
  dsp/SampleDelay.h \
  dsp/SampleDelayFunction.h \
  dsp/ScalarFilter.h \
  dsp/Shape.h \
  dsp/Simultaneous.h \
  dsp/SingleThread.h \
  dsp/SpectralKurtosis.h \
  dsp/Stats.h \
  dsp/Switcher.h \
  dsp/TFPFilterbank.h \
  dsp/TScrunch.h \
  dsp/TimeOrder.h \
  dsp/Truncate.h \
  dsp/convolution_block_constraints.h \
  dsp/filterbank_engine.h \
  dsp/on_host.h

libGeneral_la_SOURCES = \
  ACFilterbank.C \
  Apodization.C \
  AutoCorrelation.C \
  Bandpass.C \
  BandpassMonitor.C \
  ChanPolSelect.C \
  Chomper.C \
  Convolution.C \
  Derotation.C \
  Dedispersion.C \
  DedispersionHistory.C \
  DedispersionPipe.C \
  DedispersionPipeConfig.C \
  DedispersionSampleDelay.C \
  DelayStartTime.C \
  Detection.C \
  Dump.C \
  DynamicFilter.C \
  ExamplePipeline.C \
  FScrunch.C \
  FZoom.C \
  Filterbank.C \
  FilterbankBench.C \
  FilterbankConfig.C \
  FilterbankEngineCPU.C \
  FourthMoment.C \
  GeometricDelay.C \
  IncoherentFilterbank.C \
  InverseFilterbank.C \
  InverseFilterbankConfig.C \
  InverseFilterbankEngineCPU.C \
  InverseFilterbankResponse.C \
  LevelMonitor.C \
  LoadToSlice.C \
  LoadToSliceN.C \
  LoadToQuantize.C \
  Mask.C \
  MaskTimes.C \
  MultiThread.C \
  OptimalFFT.C \
  OptimalFilterbank.C \
  PlasmaResponse.C \
  PScrunch.C \
  PolnCalibration.C \
  PolnReshape.C \
  PolnSelect.C \
  RFIFilter.C \
  RFIZapper.C \
  Rescale.C \
  Resize.C \
  RescaleScaleOffsetDump.C \
  RescaleMeanStdCalculator.C \
  RescaleMedianMadCalculator.C \
  Response.C \
  ResponseProduct.C \
  SKDetector.C \
  SKFilterbank.C \
  SKMasker.C \
  SampleDelay.C \
  ScalarFilter.C \
  Shape.C \
  SingleThread.C \
  SpectralKurtosis.C \
  Stats.C \
  TFPFilterbank.C \
  TScrunch.C \
  TimeOrder.C \
  Truncate.C \
  cross_detect.c \
  cross_detect.h \
  cross_detect.ic \
  dsp_verbosity.C \
  mfilter.c \
  optimize_fft.c \
  stokes_detect.c \
  stokes_detect.h \
  stokes_detect.ic


bin_PROGRAMS = dmsmear digidiff digitxt digimon digihist digidada digislice \
  filterbank_speed fix_dspsr_K_bug

dmsmear_SOURCES = dmsmear.C
digidiff_SOURCES = digidiff.C
digitxt_SOURCES = digitxt.C
digimon_SOURCES = digimon.C
digihist_SOURCES = digihist.C
digidada_SOURCES = digidada.C
digislice_SOURCES = digislice.C
filterbank_speed_SOURCES = filterbank_speed.C
fix_dspsr_K_bug_SOURCES = fix_dspsr_K_bug.C

# bin_PROGRAMS += test_InverseFilterbankResponse test_FIRFilter
# test_InverseFilterbankResponse_SOURCES = test_InverseFilterbankResponse.C
# test_FIRFilter_SOURCES = test_FIRFilter.C


if HAVE_CUFFT

nobase_include_HEADERS += CUFFTError.h \
  dsp/ChanPolSelectCUDA.h \
  dsp/ConvolutionCUDA.h \
  dsp/ConvolutionCUDACallbacks.h \
  dsp/ConvolutionCUDASpectral.h \
  dsp/DelayStartTimeCUDA.h \
  dsp/DetectionCUDA.h \
  dsp/filterbank_cuda.h \
  dsp/FilterbankEngineCUDA.h \
  dsp/FScrunchCUDA.h \
  dsp/FZoomCUDA.h \
  dsp/InverseFilterbankEngineCUDA.h \
  dsp/PolnSelectCUDA.h \
  dsp/PScrunchCUDA.h \
  dsp/RescaleCUDA.h \
  dsp/RescaleMeanStdDevCalculatorCUDA.h \
  dsp/RescaleMedianMadCalculatorCUDA.h \
  dsp/SampleDelayCUDA.h \
  dsp/SKComputerCUDA.h \
  dsp/SKDetectorCUDA.h \
  dsp/SKFilterbankCUDA.h \
  dsp/SKMaskerCUDA.h \
  dsp/SpectralKurtosisCUDA.h \
  dsp/TScrunchCUDA.h

CUDA_SOURCES = ChanPolSelectCUDA.cu \
  ConvolutionCUDA.cu \
  ConvolutionCUDACallbacks.cu \
  ConvolutionCUDASpectral.cu \
  CUFFTError.C \
  DelayStartTimeCUDA.cu \
  DetectionCUDA.cu \
  FilterbankEngineCUDA.cu \
  FScrunchCUDA.cu \
  FZoomCUDA.cu \
  InverseFilterbankEngineCUDA.cu \
  PolnSelectCUDA.cu \
  PScrunchCUDA.cu \
  RescaleCUDA.cu \
  RescaleMeanStdDevCalculatorCUDA.cu \
  RescaleMedianMadCalculatorCUDA.cu \
  SampleDelayCUDA.cu \
  SKComputerCUDA.cu \
  SKDetectorCUDA.cu \
  SKFilterbankCUDA.cu \
  SKMaskerCUDA.cu \
  SpectralKurtosisCUDA.cu \
  TScrunchCUDA.cu

libGeneral_la_SOURCES += $(CUDA_SOURCES)

bin_PROGRAMS += fftbatch_speed
fftbatch_speed_SOURCES = fftbatch_speed.C

if HAVE_CUFFT_CALLBACKS

bin_PROGRAMS += cufft_callback_bench

cufft_callback_bench_SOURCES = cufft_callback_bench.cu

ConvolutionCUDACallbacks.lo: ConvolutionCUDACallbacks.cu
	$(top_srcdir)/config/cudalt.py $(top_builddir)/libtool $@ $(CUDA_NVCC) -dc -c $<
	$(CUDA_NVCC) -o ConvolutionCUDACallbacks_DC.o -dlink ConvolutionCUDACallbacks.o -lcufft_static

cufft_callback_bench.lo: cufft_callback_bench.cu
	$(top_srcdir)/config/cudalt.py $(top_builddir)/libtool $@ $(CUDA_NVCC) -dc -c $<
	$(CUDA_NVCC) -o cufft_callback_bench_DC.o -dlink cufft_callback_bench.o -lcufft_static

endif
endif

check_PROGRAMS = test_PolnCalibration test_OptimalFFT test_Apodization test_MaskTimes

test_PolnCalibration_SOURCES = test_PolnCalibration.C
test_OptimalFFT_SOURCES = test_OptimalFFT.C
test_Apodization_SOURCES = test_Apodization.C
test_MaskTimes_SOURCES = test_MaskTimes.C

libGeneral_la_LIBADD = @YAMLCPP_LIBS@

if HAVE_PGPLOT

  ###########################################################################
  #
  # start PGPLOT-specific code
  #

  libGeneral_la_SOURCES += ExcisionStatsPlotter.C BitStatsPlotter.C
  libGeneral_la_LIBADD += @PGPLOT_LIBS@
  bin_PROGRAMS += digistat passband

  digistat_SOURCES = digistat.C
  digistat_LDADD = @PSRPLOT_LIBS@ $(LDADD)

  passband_SOURCES = passband.C
  passband_LDADD = @PSRPLOT_LIBS@ $(LDADD)

  #
  # end PGPLOT-specific code
  #
  ###########################################################################

endif

if HAVE_fits

  ##########################################################################
  #
  # start PSRFITS-specific code
  #

  nobase_include_HEADERS += dsp/LoadToFITS.h dsp/LoadToFITSN.h
  libGeneral_la_LIBADD += @CFITSIO_LIBS@
  libGeneral_la_SOURCES += LoadToFITS.C LoadToFITSN.C

  bin_PROGRAMS += digifits
  digifits_SOURCES = digifits.C
  digifits_LDADD = @CFITSIO_LIBS@ $(LDADD)

  #
  # end PSRFITS-specific code
  #
  ##########################################################################

endif

if HAVE_sigproc

  ###########################################################################
  #
  # start sigproc-specific code
  #

  nobase_include_HEADERS += dsp/LoadToFil.h dsp/LoadToFilN.h
  libGeneral_la_SOURCES += LoadToFil.C LoadToFilN.C

  bin_PROGRAMS += digifil
  digifil_SOURCES = digifil.C


if HAVE_dada

  bin_PROGRAMS += the_decimator

  the_decimator_SOURCES = the_decimator.C
  the_decimator_LDADD = $(LDADD) @OPENSSL_LIBS@ @PSRXML_LIBS@ @PSRDADA_LIBS@
  the_decimator_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS) @PSRXML_CFLAGS@
endif

  #
  # end sigproc-specific code
  #
  ###########################################################################

endif

#############################################################################
#

SUBDIRS = testutils tests

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

LDADD = libGeneral.la \
	$(top_builddir)/Kernel/libdspbase.la \
	$(top_builddir)/Signal/Statistics/libStatistics.la \
	@PGPLOT_LIBS@ @CUDA_LIBS@ @YAMLCPP_LIBS@

AM_CPPFLAGS += @PSRDADA_CFLAGS@ @PGPLOT_CFLAGS@ @CUFFT_CFLAGS@ @CFITSIO_CFLAGS@ @YAMLCPP_CFLAGS@
AM_CXXFLAGS = @OPENMP_CFLAGS@

if HAVE_CUFFT
if HAVE_CUFFT_CALLBACKS
LDADD += ConvolutionCUDACallbacks_DC.o @CUFFT_CALLBACKS_LIBS@
else
LDADD += @CUFFT_LIBS@
endif
endif

libs: install-libLTLIBRARIES

