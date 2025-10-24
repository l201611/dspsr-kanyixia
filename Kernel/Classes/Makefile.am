
noinst_LTLIBRARIES = libClasses.la

nobase_include_HEADERS = EventEmitter.h \
	Rational.h \
	ascii_header.h \
	debug.h \
	environ.h \
	Expect.h \
	relative_path.h \
	TimeInterval.h \
	dsp/ASCIIObservation.h \
	dsp/BitSeries.h \
	dsp/BitTable.h \
	dsp/BitUnpacker.h \
	dsp/BlockFile.h \
	dsp/BlockIterator.h \
	dsp/BlockSize.h \
	dsp/BufferingPolicy.h \
	dsp/ChannelOrder.h \
	dsp/CloneArchive.h \
	dsp/CommandLineHeader.h \
    dsp/DADABlockFile.h \
	dsp/DADAFile.h \
	dsp/DADAHeader.h \
	dsp/DADAOutputFile.h \
	dsp/DataSeries.h \
	dsp/DataSource.h \
	dsp/Digitizer.h \
	dsp/DualFile.h \
	dsp/DummyFile.h \
	dsp/EightBitOne.h \
	dsp/EightBitUnpacker.h \
	dsp/ExcisionUnpacker.h \
	dsp/FIRFilter.h \
	dsp/File.h \
	dsp/FloatUnpacker.h \
	dsp/FourBitTwo.h \
	dsp/FourBitUnpacker.h \
	dsp/GenericEightBitUnpacker.h \
	dsp/GenericFourBitUnpacker.h \
	dsp/GenericSixteenBitUnpacker.h \
	dsp/GenericTwoBitCorrection.h \
	dsp/GenericVoltageDigitizer.h \
	dsp/HasInput.h \
	dsp/HasOutput.h \
	dsp/HistUnpacker.h \
	dsp/IOManager.h \
	dsp/Input.h \
	dsp/InputBuffering.h \
	dsp/InputBufferingShare.h \
	dsp/InputSource.h \
	dsp/MPIRoot.h \
	dsp/MPIServer.h \
	dsp/MPITrans.h \
	dsp/Memory.h \
	dsp/MemoryHost.h \
	dsp/MiniExtension.h \
	dsp/MiniPlan.h \
	dsp/MultiFile.h \
	dsp/Multiplex.h \
	dsp/NLowLookup.h \
	dsp/Observation.h \
	dsp/ObservationChange.h \
	dsp/ObservationInterface.h \
	dsp/Operation.h \
	dsp/OperationPerformanceMetrics.h \
	dsp/OperationThread.h \
	dsp/OutputArchive.h \
	dsp/OutputFile.h \
	dsp/OutputFileShare.h \
	dsp/ParallelBitSeries.h \
	dsp/ParallelInput.h \
	dsp/ParallelIOManager.h \
	dsp/ParallelUnpacker.h \
	dsp/PrestoObservation.h \
	dsp/Reserve.h \
	dsp/Scratch.h \
	dsp/Seekable.h \
	dsp/SerialFiles.h \
	dsp/SignalPath.h \
	dsp/SingleFile.h \
	dsp/SingleUnpacker.h \
	dsp/Sink.h \
	dsp/SixteenBitUnpacker.h \
	dsp/StepIterator.h \
	dsp/SubByteTwoBitCorrection.h \
	dsp/Source.h \
	dsp/SourceFactory.h \
	dsp/TimeSeries.h \
	dsp/TFPOffset.h \
	dsp/Transformation.h \
	dsp/TwoBit1or2.h \
	dsp/TwoBitCorrection.h \
	dsp/TwoBitCorrectionConfig.h \
	dsp/TwoBitFour.h \
	dsp/TwoBitLookup.h \
	dsp/TwoBitMask.h \
	dsp/TwoBitTable.h \
	dsp/UniversalInputBuffering.h \
	dsp/Unpacker.h \
	dsp/UnpackerIterator.h \
	dsp/UnpackerSource.h \
	dsp/WeightedTimeSeries.h \
	dsp/dsp.h \
	dsp/dspExtension.h \
	dsp/excision_unpack.h \
	dsp/infodata.h

libClasses_la_SOURCES = \
    base_verbosity.C \
    ASCIIObservation.C \
	BitSeries.C \
	BitTable.C \
	BitUnpacker.C \
	BlockFile.C \
	BlockSize.C \
	ChannelOrder.C \
	CloneArchive.C \
	CommandLineHeader.C \
    DADABlockFile.C \
	DADAFile.C \
	DADAHeader.C \
	DADAOutputFile.C \
	DataSeries.C \
	Digitizer.C \
	DualFile.C \
	DummyFile.C \
	EightBitUnpacker.C \
	ExcisionUnpacker.C \
	Expect.C \
	FIRFilter.C \
	File.C \
	FloatUnpacker.C \
	FourBitUnpacker.C \
	GenericEightBitUnpacker.C \
	GenericFourBitUnpacker.C \
	GenericSixteenBitUnpacker.C \
	GenericTwoBitCorrection.C \
	GenericVoltageDigitizer.C \
	HistUnpacker.C \
	IOManager.C \
	Input.C \
	InputBuffering.C \
	InputBufferingShare.C \
	Memory.C \
	MemoryHost.C \
	MultiFile.C \
	Multiplex.C \
	NLowLookup.C \
	Observation.C \
	ObservationChange.C \
	ObservationInterface.C \
	Operation.C \
	OperationThread.C \
	OutputFile.C \
	OutputFileShare.C \
	ParallelBitSeries.C \
	ParallelInput.C \
	ParallelIOManager.C \
	ParallelUnpacker.C \
	PrestoObservation.C \
	Rational.C \
	relative_path.C \
	Reserve.C \
	Scratch.C \
	Seekable.C \
	SerialFiles.C \
	SingleFile.C \
	SingleUnpacker.C \
	SignalPath.C \
	SixteenBitUnpacker.C \
	SourceFactory.C \
	SubByteTwoBitCorrection.C \
	TimeInterval.C \
	TimeSeries.C \
	TwoBit1or2.C \
	TwoBitCorrection.C \
	TwoBitCorrectionConfig.C \
	TwoBitFour.C \
	TwoBitLookup.C \
	TwoBitTable.C \
	Unpacker.C \
	UnpackerIterator.C \
	Unpacker_create.C \
	WeightedTimeSeries.C \
	ascii_header.c \
	dsp.C \
	dspExtension.C

if HAVE_CUDA
nobase_include_HEADERS += \
  dsp/GenericEightBitUnpackerCUDA.h \
  dsp/GenericVoltageDigitizerCUDA.h \
  dsp/FloatUnpackerCUDA.h \
  dsp/LaunchConfig.h \
  dsp/MemoryCUDA.h \
  dsp/TimeSeriesCUDA.h \
  dsp/TransferBitSeriesCUDA.h \
  dsp/TransferCUDA.h


CUDA_SOURCES = \
  check_error.C \
  GenericEightBitUnpackerCUDA.cu \
  GenericVoltageDigitizerCUDA.cu \
  FloatUnpackerCUDA.cu \
  LaunchConfig.C \
  MemoryCUDA.C \
  TimeSeriesCUDA.cu \
  TransferBitSeriesCUDA.C \
  TransferCUDA.C

libClasses_la_SOURCES += $(CUDA_SOURCES)
libClasses_la_LIBADD = @CUDA_LIBS@
endif


check_PROGRAMS = test_BlockIterator test_environ

test_BlockIterator_SOURCES = test_BlockIterator.C

#############################################################################
#

SUBDIRS = testutils tests

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

LDADD = libClasses.la

AM_CPPFLAGS += @CUDA_CFLAGS@ @PSRDADA_CFLAGS@

NOW_IN_EPSIC := true_math.h

all-local:
	for file in $(NOW_IN_EPSIC); do \
	rm -f $(top_builddir)/local_include/$$file ; done

