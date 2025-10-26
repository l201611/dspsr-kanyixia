# SWIN_LIB_MARK5ACCESS([ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]])
# ----------------------------------------------------------
AC_DEFUN([SWIN_LIB_MARK5ACCESS],
[
  AC_PROVIDE([SWIN_LIB_MARK5ACCESS])

  AC_ARG_WITH([mark5access-dir],
              AC_HELP_STRING([--with-mark5access-dir=DIR],
                             [MARK5ACCESS is installed in DIR]))

  MARK5ACCESS_CFLAGS=""
  MARK5ACCESS_LIBS=""

  if test x"$with_mark5access_dir" = xno; then
    # user disabled mark5access. Leave cache alone.
    have_mark5access="User disabled mark5access."
  else

    AC_MSG_CHECKING([for mark5access installation])

    # "yes" is not a specification
    if test x"$with_mark5access_dir" = xyes; then
      with_mark5access_dir=
    fi

    have_mark5access="not found"

    ac_save_CPPFLAGS="$CPPFLAGS"
    ac_save_LIBS="$LIBS"

    if test x"$with_mark5access_dir" != x; then
      mark5access_inc=""
      for dir in "$with_mark5access_dir/include" \
                 "$with_mark5access_dir/include/mark5access" \
                 "$with_mark5access_dir"; do
        if test -d "$dir"; then
          mark5access_inc="$dir"
          break
        fi
      done

      if test x"$mark5access_inc" != x; then
        MARK5ACCESS_CFLAGS="-I$mark5access_inc $MARK5ACCESS_CFLAGS"
      fi

      mark5access_libdir=""
      for dir in "$with_mark5access_dir/lib" \
                 "$with_mark5access_dir/lib64" \
                 "$with_mark5access_dir"; do
        if test -d "$dir"; then
          mark5access_libdir="$dir"
          MARK5ACCESS_LIBS="-L$dir $MARK5ACCESS_LIBS"
          break
        fi
      done

      MARK5ACCESS_LIBS="$MARK5ACCESS_LIBS -lmark5access"

      CPPFLAGS="$MARK5ACCESS_CFLAGS $CPPFLAGS"
      LIBS="$MARK5ACCESS_LIBS $LIBS"

      AC_TRY_LINK([#include <mark5access.h>], [new_mark5_stream(0,0);],
                  have_mark5access=yes, have_mark5access=no)

      if test $have_mark5access != yes; then
        MARK5ACCESS_CFLAGS=""
        MARK5ACCESS_LIBS=""
      fi
    else
      pkgc_mark5access_cflags=`pkg-config --cflags mark5access 2>/dev/null`
      pkgc_mark5access_libs=`pkg-config --libs mark5access 2>/dev/null`

      CPPFLAGS="$pkgc_mark5access_cflags $CPPFLAGS"
      LIBS="$pkgc_mark5access_libs $LIBS"

      AC_TRY_LINK([#include <mark5access.h>], [new_mark5_stream(0,0);],
                  have_mark5access=yes, have_mark5access=no)

      if test $have_mark5access = yes; then
        MARK5ACCESS_CFLAGS="$pkgc_mark5access_cflags"
        MARK5ACCESS_LIBS="$pkgc_mark5access_libs"
      fi
    fi

    LIBS="$ac_save_LIBS"
    CPPFLAGS="$ac_save_CPPFLAGS"

  fi

  AC_MSG_RESULT([$have_mark5access])

  if test "$have_mark5access" = "yes"; then
    AC_DEFINE([HAVE_MARK5ACCESS], [1], [Define if the mark5access library is present])
    [$1]
  else
    AC_MSG_NOTICE([Ensure that the PKG_CONFIG_PATH environment variable points to])
    AC_MSG_NOTICE([the lib/pkgconfig sub-directory of the root directory where])
    AC_MSG_NOTICE([the mark5access library was installed.])
    AC_MSG_NOTICE([Alternatively, use the --with-mark5access-dir option.])
    [$2]
  fi

  AC_SUBST(MARK5ACCESS_LIBS)
  AC_SUBST(MARK5ACCESS_CFLAGS)
  AM_CONDITIONAL(HAVE_MARK5ACCESS,[test "$have_mark5access" = "yes"])

])

