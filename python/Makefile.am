
nodist_python_PYTHON = dspsr.py

python_LTLIBRARIES = _dspsr.la

_dspsr_la_SOURCES = dspsr.i

nodist__dspsr_la_SOURCES = ./dspsr_wrap.cxx

_dspsr_la_LDFLAGS = -module -avoid-version -no-undefined

_dspsr_la_LIBADD = \
		   $(top_builddir)/Signal/libdspsr.la \
		   $(top_builddir)/Signal/libdspdsp.la \
		   $(top_builddir)/Signal/libdspstats.la \
		   $(top_builddir)/Kernel/libdspbase.la \
		   @CUFFT_LIBS@ @CUDA_LIBS@ @PYTHON_LDFLAGS@

SWIG_FILE = $(srcdir)/dspsr.i
SWIG_FLAGS = -Wall -python -I$(top_builddir)/local_include -I$(top_builddir) @PSRCHIVE_INCLUDE@
SWIG_DEPS = $(DEPDIR)/swig.deps

dspsr_wrap.cxx: $(SWIG_FILE) $(top_builddir)/config.h
	$(SWIG) $(SWIG_FLAGS) -outcurrentdir $(srcdir)/dspsr.i

#############################################################################
#
# regenerate dspsr_wrap.cxx and dspsr.py if C++ headers change
#

$(SWIG_DEPS): $(SWIG_FILE)
	$(SWIG) $(SWIG_FLAGS) $(DEFS) -MM -outcurrentdir $(SWIG_FILE) > $(SWIG_DEPS)

-include $(SWIG_DEPS)

CLEANFILES = dspsr.py dspsr.pyc dspsr_wrap.cxx $(SWIG_DEPS)

#############################################################################
#

AM_CPPFLAGS = @SWIG_PYTHON_CPPFLAGS@ -I$(top_builddir)/local_include @PSRCHIVE_INCLUDE@
