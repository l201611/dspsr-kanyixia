
SUBDIRS = Classes Formats . Applications

# this convenience library is not installed and therefore can be used for linking unit tests
noinst_LTLIBRARIES = libdspbase_local.la
libdspbase_local_la_SOURCES =
libdspbase_local_la_LIBADD = Classes/libClasses.la Formats/libFormats.la

lib_LTLIBRARIES = libdspbase.la
libdspbase_la_SOURCES  =
libdspbase_la_LIBADD = libdspbase_local.la

include $(top_srcdir)/config/Makefile.local
