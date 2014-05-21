Overview
=====

VPython is written primarily in C++ and compiled with the g++
compiler.  It does have a number of components from other projects.


Glossary
----

Developing VPython uses a number of different tools, not all of which will be familiar to all developers.

[**Boost**](http://www.boost.org): a standard set of C++ libraries.  Most
Boost libraries consist of inline functions and templates in header files.

[**Enthought**](https://enthought.com/products/epd/):  a company selling to scientific Python users.   EPD (Enthought Python Distribution) and Canopy (desktop environment) have free and commericial versions.

[**GCC**](http://gcc.gnu.org): Gnu Compiler Collection, which includes the
*gcc* compiler and *gcov* coverage/profiler tool.

[**GStreamer**](http://gstreamer.freedesktop.org): An open multimedia
library for contructing, chaining, and playing media.

[**GTK+**](http://www.gtk.org):  a multi-platform GUI and widget toolkit, sometimes known as *the GIMP Toolkit*.  Related libraries include *gtkmm* (C++ interfaces for GTK+ and Gnome), *GtkGLExt* (extensions for OpenGL), and *GtkGLExtMM* (C++ interface to *GtkGLExt*).

[**Markdown**](https://daringfireball.net/projects/markdown/): A typesetter
format that produces *html* files from *md* files.  Note that GitHub
automatically renders markdown files.

[**NumPy**](http://www.numpy.org): a Python package for scientific
computing.  It includes N-dimensional array objects, good random numbers,
and linear algebra tools.

[**OpenGL**](http://www.opengl.org):  A cross-platform, multi-language API for 2D and 3D vector graphics.

[**Polygon 2**](http://www.j-raedler.de/projects/polygon/):  A 2D polygon package with useful operators and bindings to the [General Polygon Clipping Library (GPC)](http://www.cs.man.ac.uk/~toby/gpc/).  Has a complex license.

[**Travis CI**](https://travis-ci.org): a hosted continuous
  integration service that builds and runs tests according to the
  configuration in *./.travis.yml*.  A build is initiated for each
  commit or pull-request posted to GitHub.  Each build creates a
  [log](https://travis-ci.org/BruceSherwood/vpython-wx), updates a
  [small graphic](https://travis-ci.org/BruceSherwood/vpython-wx.png?branch=master)
  at the top left of the main project's
  [README.md](https://github.com/BruceSherwood/vpython-wx/blob/master/README.md),
  and emails the author of the commit if any tests fail.

[**WxPython**](http://www.wxpython.org): a cross-platform Python
library for the Wx GUI and widget kit.  It blends Python with wrappers
for the [wxWidgets](http://docs.wxwidgets.org/3.0/index.html) C++
class libraries.  This library compares with the
[TkInter](https://docs.python.org/2/library/tkinter.html) library
wrapping [Tcl/Tk](https://www.tcl.tk) and either
[PySide](https://pypi.python.org/pypi/PySide) or
[PyQt](http://www.riverbankcomputing.com/software/pyqt/intro) wrapping
[Qt](http://qt-project.org).  Selection among GUI libraries is often
made based on the licenses.

[**xvfb**](http://en.wikipedia.org/wiki/Xvfb): A minimal virtual
framebuffer for X Windows that is used in testing.

File System Orientation
----

**README.md**: user readme file in markdown format, also used as the
  project discription on GitHub.

**DEVELOP.md**:  This file; information for new VPython developers.

The rest of the files in the root directory are related to builds.
These include:

   * **INSTALL.TXT**:  documentation for building on Linux.
   * **MAC-OSX.TXT**:  documentation for building on OS/X 10.6+
   * **MSWINDOWS.TXT**: obsolete documentation for building on
     MS-Windows.  See *VCBuild/VCBuild.txt* instead.
   * **compilevisual.py**: a program which will force creation of .pyc
     files for Visual modules.
   * **MakeVPython\*.iss**: Inno Setup configurations for bundling
     MS-Windows versions.
   * **.travis.yml**: the configuration file for Travis CI, which builds
     on a generic Linux machine on each GitHub commit.

These are the subdirectories off the root:

   * **include/**:  C++ header files.
   * **src/**: a makefile and C++ source files
   * **tests/**:  contains a simple integration test
   * **site-packages/**:  the visual and vidle Python modules.
   * **VCBuild/**:  build files for MS-Windows.


License Overview
---

VPython consists of a number of components with distinct licenses.
*The Visual Library* uses a simple, but custom, attribution license.
*The Polygon Library* contains two licenses: the *Lesser Gnu Public
Library (LGPL)* of an unspecified version and an additional license
for the incorporated *Generic Polygon Library* which prohibits
commericial use without an additional license.  The *num_util.\**
files use the derivative *Boost Software License* while
*site-packages/visual_common/shapes.py* is released under the *Blender
Artistic License* which requires some documentation of changes.
*NumPy* uses the attribution *NumPy License*.  There are likely other
licenses lurking about as well.

In short, it should be fine to use for non-commerical use.
