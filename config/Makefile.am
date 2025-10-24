
#############################################################################
#
# This Makefile smooths over various bugs and bumps and other features
# that break the other GNU autotools-generated Makefiles
#

EXTRA_DIST = backends.default formats.sh Makefile.aclocal cudalt.py

all: aclocal_update

libs:

MAKEFILE_ACLOCAL = $(top_srcdir)/config/Makefile.aclocal

include $(MAKEFILE_ACLOCAL)

$(MAKEFILE_ACLOCAL): $(PSRCHIVE_ACLOCAL)/Makefile.aclocal
	cp $(PSRCHIVE_ACLOCAL)/Makefile.aclocal $(MAKEFILE_ACLOCAL)

$(PSRCHIVE_ACLOCAL)/Makefile.aclocal:
	@echo; echo; echo; \
	echo A required configuration file is missing; \
	echo Please update your PSRCHIVE installation; \
	echo; echo; echo; \
	exit -1

nodist_include_HEADERS = dspsr_version.h
.PHONY: dspsr_version.h

dspsr_version.h:
	GIT_HASH=`cd $(top_srcdir); git describe --always --dirty --match 'NOT A TAG'`; echo "#define DSPSR_COMMIT_HASH \"$$GIT_HASH\"" > dspsr_version.tmp
	if diff -q dspsr_version.tmp dspsr_version.h >/dev/null 2>&1; then rm dspsr_version.tmp; else mv dspsr_version.tmp dspsr_version.h; fi

CLEANFILES = dspsr_version.h

