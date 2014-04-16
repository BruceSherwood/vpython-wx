Overview
=====

VPython is written primarily in C++ and compiled with the g++
compiler.  It does have a number of components from other projects.


Glossary
===

Developing VPython uses a number of different tools, not all of which will be familiar to all developers.

[**Boost**](http://www.boost.org): a standard set of C++ libraries.  Most
Boost libraries consist of inline functions and templates in header files.

[**GCC**](http://gcc.gnu.org): Gnu Compiler Collection, which includes the
*gcc* compiler and *gcov* coverage/profiler tool.

[**GStreamer**](http://gstreamer.freedesktop.org): An open multimedia
library for contructing, chaining, and playing media.

[**Markdown**](https://daringfireball.net/projects/markdown/): A typesetter
format that produces *html* files from *md* files.  Note that GitHub
automatically renders markdown files.

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
