if HAVE_PGPLOT

bin_PROGRAMS = searchplot

searchplot_SOURCES = searchplot.C
searchplot_LDADD = $(LDADD) @PGPLOT_LIBS@ @CUFFT_LIBS@

LDADD = $(top_builddir)/More/Plotting/libPlotting.la \
	$(top_builddir)/Signal/libdspdsp.la \
	$(top_builddir)/Signal/libdspstats.la \
	$(top_builddir)/Kernel/libdspbase.la \
	@CUFFT_LIBS@ @OPENMP_CFLAGS@

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @PGPLOT_CFLAGS@ @CFITSIO_CFLAGS@ @CUFFT_CFLAGS@

endif
