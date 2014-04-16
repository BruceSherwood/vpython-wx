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

[**Markdown**](https://daringfireball.net/projects/markdown/): A typesetter
format that produces *html* files from *md* files.  Note that GitHub
automatically renders *.md* files.

[**Polygon 2**](http://www.j-raedler.de/projects/polygon/):  A 2D polygon package with useful operators and bindings to the General Polygon Clipping Library.  Has a complex license.

[**Travis CI**](https://travis-ci.org): a hosted continuous
  integration service that builds and runs tests according to the
  configuration in *./.travis.yml*.  A build is initiated for each
  commit or pull-request posted to GitHub.  Each build creates a
  [log](https://travis-ci.org/BruceSherwood/vpython-wx), updates a
  [small graphic](https://travis-ci.org/BruceSherwood/vpython-wx.png?branch=master)
  at the top left of the main project's
  [README.md](https://github.com/BruceSherwood/vpython-wx/blob/master/README.md),
  and emails the author of the commit if any tests fail.

[**WxPython**](http://www.wxpython.org): a cross-platform GUI and
widget kit blending Python with [C++ class libraries]
(http://docs.wxwidgets.org/3.0/index.html).  This library compares
with TkInter for Tcl/Tk or PySide and PyQt for Qt. Many people select
among this list based on their licenses.

[**xvfb**](http://en.wikipedia.org/wiki/Xvfb): A minimal virtual
framebuffer for X Windows.  Useful for testing.
