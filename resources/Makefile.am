
GTEST_DIR = $(srcdir)/gtest/googletest
GMOCK_DIR = $(srcdir)/gtest/googlemock

GTEST_SRC = $(GTEST_DIR)/src/gtest-all.cc
GMOCK_SRC = $(GMOCK_DIR)/src/gmock-all.cc

check_LTLIBRARIES = libgtest.la
nodist_libgtest_la_SOURCES = gtest-all.cc gmock-all.cc
libgtest_la_CPPFLAGS = -I$(GTEST_DIR)/include -I$(GTEST_DIR) -I$(GMOCK_DIR)/include -I$(GMOCK_DIR)
libgtest_la_LDFLAGS = -pthread

gtest-all.cc: $(GTEST_SRC)
	cp $(GTEST_SRC) .

gmock-all.cc: $(GMOCK_SRC)
	cp $(GMOCK_SRC) .

BUILT_SOURCES = gtest-all.cc gmock-all.cc
CLEANFILES = gtest-all.cc gmock-all.cc
EXTRA_DIST = $(GTEST_DIR)/src \
  $(GTEST_DIR)/include \
  $(GMOCK_DIR)/src \
  $(GMOCK_DIR)/include

$(GTEST_SRC):
	echo
	echo "Missing googletest submodule; to fix, run"
	echo
	echo "cd $(abs_top_srcdir) && git submodule update --init --recursive"
	echo

$(GMOCK_SRC):
	echo
	echo "Missing googletest submodule; to fix, run"
	echo
	echo "cd $(abs_top_srcdir) && git submodule update --init --recursive"
	echo
