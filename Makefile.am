## Process this file with automake to produce Makefile.in
# Makefile for the top-level source directory

AUTOMAKE_OPTIONS = foreign dist-bzip2

SUBDIRS = site-packages/visual site-packages/vis src

# Logic to distribute the header files and miscellaneous files.
EXTRA_DIST = include HACKING.txt INSTALL.txt \
	authors.txt license.txt NEWS.txt \
	dependencies/Readme.txt dependencies/threadpool
dist-hook:
	rm -rf `find $(distdir)/include -name CVS` $(distdir)/include/config.h

# Logic to install the "vpython" script
bin_SCRIPTS = bin/vpython

if BUILD_EXAMPLES
SUBDIRS += examples
endif

##if BUILD_VIDLE
##SUBDIRS += site-packages/vidle
##endif

if BUILD_DOCS
SUBDIRS += docs
endif

