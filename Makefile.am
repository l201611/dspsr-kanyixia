
SUBDIRS	= config resources Kernel Signal Management test

if HAVE_PYTHON
if HAVE_SWIG
SUBDIRS += python
endif
endif

if HAVE_PGPLOT
SUBDIRS += More
endif

include $(top_srcdir)/config/Makefile.root

TESTS = \
  Kernel/Classes/test_BlockIterator \
  Signal/General/test_PolnCalibration \
  Signal/General/test_OptimalFFT \
  Signal/General/test_Apodization \
  Signal/Statistics/test_Romberg

# Note that any test defined in Kernel/Formats/<dir> should be added
# only if HAVE_<dir>, as done for HAVE_dada and HAVE_ska1 below
TESTS += \
  Kernel/Classes/tests/GenericVoltageDigitizerTest \
  Kernel/Classes/tests/GaussianNoiseSourceTest \
  Kernel/Classes/tests/ImpulsiveNoiseSourceTest \
  Kernel/Classes/tests/SumSourceTest \
  Kernel/Classes/tests/WeightedTimeSeriesTest \
  Signal/General/tests/ChanPolSelectTest \
  Signal/General/tests/ConvolutionTest \
  Signal/General/tests/InverseFilterbankTest \
  Signal/General/tests/LoadToQuantizeTest \
  Signal/General/tests/RescaleTest \
  Signal/General/tests/ResponseTest \
  Signal/General/tests/ResponseProductTest \
  Signal/General/tests/RescaleMedianMadCalculatorTest \
  Signal/General/tests/TScrunchTest \
  Signal/Pulsar/tests/CyclicFoldTest

if HAVE_PSRDADA
TESTS += Kernel/Classes/tests/DADAOutputFileTest
if HAVE_dada
TESTS += Kernel/Formats/dada/tests/OutputDADABufferTest
endif
endif

if HAVE_ska1
TESTS += Kernel/Formats/ska1/tests/SKAParallelUnpackerTest
endif

ACLOCAL_AMFLAGS = -I config

# make clean deletes the local_include directory (and any out-of-date headers)
clean-local: clean-local-include
.PHONY: clean-local-include
clean-local-include:
	-rm -rf local_include

CLEANFILES = top_hat.dat tukey.dat bartlett.dat welch.dat hanning.dat

# delete backends.list if it is an unmodified copy of config/backends.default
distclean-local:
	-if diff $(top_srcdir)/config/backends.default backends.list >/dev/null 2>&1; then rm -f backends.list; fi

