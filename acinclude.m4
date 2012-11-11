
# I have modified the following code from the stock version found in Automake
# 1.9.3, to translate backslashes into forward slashes for Window's sake.

# Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004
# Free Software Foundation, Inc.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
# 02111-1307, USA.

# AM_PATH_PYTHON([MINIMUM-VERSION], [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])

# Adds support for distributing Python modules and packages.  To
# install modules, copy them to $(pythondir), using the python_PYTHON
# automake variable.  To install a package with the same name as the
# automake package, install to $(pkgpythondir), or use the
# pkgpython_PYTHON automake variable.

# The variables $(pyexecdir) and $(pkgpyexecdir) are provided as
# locations to install python extension modules (shared libraries).
# Another macro is required to find the appropriate flags to compile
# extension modules.

# If your package is configured with a different prefix to python,
# users will have to add the install directory to the PYTHONPATH
# environment variable, or create a .pth file (see the python
# documentation for details).

# If the MINIMUM-VERSION argument is passed, AM_PATH_PYTHON will
# cause an error if the version of python installed on the system
# doesn't meet the requirement.  MINIMUM-VERSION should consist of
# numbers and dots only.

AC_DEFUN([AM_PATH_PYTHON],
 [
 AC_REQUIRE([VISUAL_CHECK_PLATFORM])
  dnl Find a Python interpreter.  Python versions prior to 1.5 are not
  dnl supported because the default installation locations changed from
  dnl $prefix/lib/site-python in 1.4 to $prefix/lib/python1.5/dist-packages
  dnl in 1.5.
  m4_define([_AM_PYTHON_INTERPRETER_LIST],
            [python python3.1 python2.7 python2.6 python2.5 python2.4])

  m4_if([$1],[],[
    dnl No version check is needed.
    # Find any Python interpreter.
    if test -z "$PYTHON"; then
      PYTHON=:
      AC_PATH_PROGS([PYTHON], _AM_PYTHON_INTERPRETER_LIST)
    fi
    am_display_PYTHON=python
  ], [
    dnl A version check is needed.
    if test -n "$PYTHON"; then
      # If the user set $PYTHON, use it and don't search something else.
      AC_MSG_CHECKING([whether $PYTHON version >= $1])
      AM_PYTHON_CHECK_VERSION([$PYTHON], [$1],
			      [AC_MSG_RESULT(yes)],
			      [AC_MSG_ERROR(too old)])
      am_display_PYTHON=$PYTHON
    else
      # Otherwise, try each interpreter until we find one that satisfies
      # VERSION.
      AC_CACHE_CHECK([for a Python interpreter with version >= $1],
	[am_cv_pathless_PYTHON],[
	for am_cv_pathless_PYTHON in _AM_PYTHON_INTERPRETER_LIST none; do
	  test "$am_cv_pathless_PYTHON" = none && break
	  AM_PYTHON_CHECK_VERSION([$am_cv_pathless_PYTHON], [$1], [break])
	done])
      # Set $PYTHON to the absolute path of $am_cv_pathless_PYTHON.
      if test "$am_cv_pathless_PYTHON" = none; then
	PYTHON=:
      else
        AC_PATH_PROG([PYTHON], [$am_cv_pathless_PYTHON])
      fi
      am_display_PYTHON=$am_cv_pathless_PYTHON
    fi
  ])

  if test "$PYTHON" = :; then
  dnl Run any user-specified action, or abort.
    m4_default([$3], [AC_MSG_ERROR([no suitable Python interpreter found])])
  else

  dnl Query Python for its version number.  Getting [:3] seems to be
  dnl the best way to do this; it's what "site.py" does in the standard
  dnl library.

  AC_CACHE_CHECK([for $am_display_PYTHON version], [am_cv_python_version],
    [am_cv_python_version=`$PYTHON -c "import sys; print(sys.version[[:3]])"`])
  AC_SUBST([PYTHON_VERSION], [$am_cv_python_version])

  dnl At times (like when building shared libraries) you may want
  dnl to know which OS platform Python thinks this is.

  AC_CACHE_CHECK([for $am_display_PYTHON platform], [am_cv_python_platform],
    [am_cv_python_platform=`$PYTHON -c "import sys; print(sys.platform)"`])
  AC_SUBST([PYTHON_PLATFORM], [$am_cv_python_platform])

  dnl Use the values of $prefix and $exec_prefix for the corresponding
  dnl values of PYTHON_PREFIX and PYTHON_EXEC_PREFIX.  These are made
  dnl distinct variables so they can be overridden if need be.  However,
  dnl general consensus is that you shouldn't need this ability.

  if test $PYTHON_PLATFORM = "win32"; then
    AC_SUBST([PYTHON_PREFIX], [`$PYTHON -c "from distutils import sysconfig; print(sysconfig.PREFIX)"`])
    AC_SUBST([PYTHON_EXEC_PREFIX], [`$PYTHON -c "from distutils import sysconfig; print(sysconfig.EXEC_PREFIX)"`])
  else
    AC_SUBST([PYTHON_PREFIX], ['${prefix}'])
    AC_SUBST([PYTHON_EXEC_PREFIX], ['${exec_prefix}'])
  fi

  dnl Set up 4 directories:

  dnl pythondir -- where to install python scripts.  This is the
  dnl   dist-packages directory, not the python standard library
  dnl   directory like in previous automake betas.  This behavior
  dnl   is more consistent with lispdir.m4 for example.
  dnl Query distutils for this directory.  distutils does not exist in
  dnl Python 1.5, so we fall back to the hardcoded directory if it
  dnl doesn't work.
  AC_CACHE_CHECK([for $am_display_PYTHON script directory],
    [am_cv_python_pythondir],
    [am_cv_python_pythondir=`$PYTHON -c "from distutils import sysconfig; print(sysconfig.get_python_lib(0,0,prefix='$PYTHON_PREFIX').replace('\\\\\','/'))" 2>/dev/null || 
    	echo "$PYTHON_PREFIX/lib/python$PYTHON_VERSION/dist-packages"`])
  AC_SUBST([pythondir], [$am_cv_python_pythondir])

  dnl pkgpythondir -- $PACKAGE directory under pythondir.  Was
  dnl   PYTHON_SITE_PACKAGE in previous betas, but this naming is
  dnl   more consistent with the rest of automake.

  AC_SUBST([pkgpythondir], [\${pythondir}/$PACKAGE])

  dnl pyexecdir -- directory for installing python extension modules
  dnl   (shared libraries)
  dnl Query distutils for this directory.  distutils does not exist in
  dnl Python 1.5, so we fall back to the hardcoded directory if it
  dnl doesn't work.
  dnl Change to put cvisualmodule in vis folder
  dnl AC_CACHE_CHECK([for $am_display_PYTHON extension module directory],
  dnl   [am_cv_python_pyexecdir],
  dnl   [am_cv_python_pyexecdir=`$PYTHON -c "from distutils import sysconfig; print(sysconfig.get_python_lib(1,0,prefix='$PYTHON_EXEC_PREFIX').replace('\\\\\','/'))" 2>/dev/null ||
  dnl      echo "${PYTHON_EXEC_PREFIX}/lib/python${PYTHON_VERSION}/dist-packages"`])
  dnl AC_SUBST([pyexecdir], [$am_cv_python_pyexecdir])
  
  AC_SUBST([pyexecdir], [\${pythondir}/vis])
  
  AC_SUBST([pkgpyexecdir], [\${pythondir}/$PACKAGE])

  dnl Run any user-specified action.
  $2
  fi

])

# A function that determins if we are on Windows or OSX, based on the host.
# Copied from the official gtk+-2 configure.in
AC_DEFUN([VISUAL_CHECK_PLATFORM],
[
	AC_MSG_CHECKING([for some Win32 platform])
	case "$host" in
	  *-*-mingw*|*-*-cygwin*)
	    platform_win32=yes
	    ;;
	  *)
	    platform_win32=no
	    ;;
	esac
	AC_MSG_RESULT([$platform_win32])
	
	AC_MSG_CHECKING([for some Mac OSX platform])
	case "$host" in
	  *-apple-darwin*)
	    platform_osx=yes
	    ;;
	  *)
	    platform_osx=no
	    ;;
	esac
	AC_MSG_RESULT([$platform_osx])
])

AC_DEFUN([VISUAL_DOCS],
[
	AC_REQUIRE([AM_PATH_PYTHON])
	AC_ARG_ENABLE([docs],
		AC_HELP_STRING([--disable-docs], [do not install html documentation]),
		[visual_build_docs=$enableval],
		[visual_build_docs="yes"])
	
	AC_ARG_WITH([html-dir],
		AC_HELP_STRING([--with-html-dir=PATH], [path to install html documentation default=pkgpythondir/docs]),
		[visual_htmldir=$withval],
		[visual_htmldir=""])		
	
	AC_MSG_CHECKING( where to install documentation)
	
	if test "x$visual_htmldir" = "x" ; then
		visualdocdir=${pythondir}/$PACKAGE/docs
	else
		visualdocdir=$visual_htmldir
	fi
	AC_SUBST( visualdocdir)
	AC_MSG_RESULT( $visualdocdir)

	AC_MSG_CHECKING( whether to install html documentation)
	AM_CONDITIONAL([BUILD_DOCS], test $visual_build_docs = "yes")
	AC_MSG_RESULT( $visual_build_docs)

])

AC_DEFUN([VISUAL_NUMERICLIBS],
[
	AC_REQUIRE([AM_PATH_PYTHON])
	
	PY_CHECK_MOD( [numpy], [array], 
	[
		AC_DEFINE([VISUAL_HAVE_NUMPY], [1])
		visual_have_numpy="yes"
		numpyincludedir=${pythondir}/numpy/core/include
	], [visual_have_numpy="no"])
	
	if test $visual_have_numpy = "no"; then
		AC_MSG_ERROR( [The numpy module could not be found but is required. See numpy.sourceforge.net for downloads.])
	fi
])

AC_DEFUN([VISUAL_VIS],
[
	AC_REQUIRE([AM_PATH_PYTHON])
	
	AC_MSG_CHECKING( where to install vis components)

    visualvisdir=${pythondir}/vis
	
	AC_MSG_RESULT( $visualvisdir)
	AC_SUBST( visualvisdir)
])

AC_DEFUN([VISUAL_EXAMPLES],
[
	AC_REQUIRE([AM_PATH_PYTHON])
	AC_ARG_ENABLE([examples],
		AC_HELP_STRING([--disable-examples], [do not install example programs]),
		[visual_build_examples=$enableval],
		[visual_build_examples="yes"])
	
	AC_ARG_WITH([example-dir],
		AC_HELP_STRING([--with-example-dir=PATH], [path to install demo programs default=pkgpythondir/examples]),
		[visual_exampledir=$withval],
		[visual_exampledir=""])		
	
	AC_MSG_CHECKING( where to install example programs)
	
	if test "x$visual_exampledir" = "x" ; then
		visualexampledir=${pythondir}/$PACKAGE/examples
	else
		visualexampledir=$visual_exampledir
	fi
	
	AC_MSG_RESULT( $visualexampledir)
	AC_SUBST( visualexampledir)
	
	AC_MSG_CHECKING( whether to install example programs)
	AM_CONDITIONAL([BUILD_EXAMPLES], test $visual_build_examples = "yes")
	AC_MSG_RESULT( $visual_build_examples)
])


dnl Modified from pyautoconf to use the Automake-supplied macro AM_PATH_PYTHON
dnl ----------------------------------------------------------------------
dnl These functions are used similar to AC_CHECK_LIB and associates.

dnl PY_CHECK_MOD(MODNAME [,SYMBOL [,ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]]])
dnl Check if a module containing a given symbol is visible to python.
AC_DEFUN([PY_CHECK_MOD],
[
	AC_REQUIRE([AM_PATH_PYTHON])
	py_mod_var=`echo $1['_']$2 | sed 'y%./+-%__p_%'`
	AC_MSG_CHECKING(for ifelse([$2],[],,[$2 in ])python module $1)
	AC_CACHE_VAL(py_cv_mod_$py_mod_var, [
	if $PYTHON -c 'import $1 ifelse([$2],[],,[; $1.$2])' 1>&AC_FD_CC 2>&AC_FD_CC; then
	  eval "py_cv_mod_$py_mod_var=yes"
	else
	  eval "py_cv_mod_$py_mod_var=no"
	fi
	])
	py_val=`eval "echo \`echo '$py_cv_mod_'$py_mod_var\`"`
	if test "x$py_val" != xno; then
	  AC_MSG_RESULT(yes)
	  ifelse([$3], [],, [$3
	])dnl
	else
	  AC_MSG_RESULT(no)
	  ifelse([$4], [],, [$4
	])dnl
	fi
])

dnl a macro to check for ability to create python extensions
dnl  AM_CHECK_PYTHON_HEADERS([ACTION-IF-POSSIBLE], [ACTION-IF-NOT-POSSIBLE])
dnl function also defines PYTHON_INCLUDES
AC_DEFUN([AM_CHECK_PYTHON_HEADERS],
[
	AC_REQUIRE([AM_PATH_PYTHON])
	AC_MSG_CHECKING(for headers required to compile python extensions)
	dnl deduce PYTHON_INCLUDES (modified by Jonathan Brandmeyer to get the info
	dnl directly from Python itself).
	dnl The following loses backslashes on Windows and gives -Ic:Python25include,
	dnl but src/Makefile.in does produce the right include statement for Windows.
	PYTHON_INCLUDES=-I`$PYTHON -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"`
	PYTHON_INCLUDES="$PYTHON_INCLUDES -I$numpyincludedir"
	AC_SUBST(PYTHON_INCLUDES)
	dnl check if the headers exist:
	save_CPPFLAGS="$CPPFLAGS"
	CPPFLAGS="$CPPFLAGS $PYTHON_INCLUDES"
	AC_TRY_CPP([#include <Python.h>],dnl
	[AC_MSG_RESULT(found)
	$1],dnl
	[AC_MSG_RESULT(not found)
	$2])
	CPPFLAGS="$save_CPPFLAGS"
])
