HOW TO BUILD VISUAL 6 USING wxPython
====

Install XCode
-------------

* If you're running OSX 10.6 (Snow Leopard):

  Install XcodeTools from the optional materials on the Mac installation DVD.
Installing XcodeTools automatically installs the large number of individual
components also listed on the DVD.

* If you're running OSX 10.7 or 10.8 (Lion, or Mountain Lion)

  Get XCode from the Mac App store (free).

* If you want to Enthought or python.org's python3.2

  You need to get XCode 3.2 from the Apple developer website.

  To build Enthought or python3.2 versions you'll also need to make a
link called 'Developer' from the root of your filesystem to wherever
you've installed XCode.  On my system, that command is `$ ln -s
/Developer-3.2.5 /Developer`.  This is because python.org python3.2
and enthought python2.7 both have `-isysroot` arguments on all their
generated compiler command line strings. This flag tells gcc where to
go to find SDKs, headers, etc..

  If you do this, you'll also want to add `/Developer/usr/bin` to your
path so that you get the right version of gcc for your python build.

* Building for Python 2.7

   If you're building for python.org's python2.7 you can use a more
modern version of Xcode (e.g., 4.6) and no link is required, and no
path change is needed.  (We'll see what happens when they release
Xcode 5!)

Getting Python
-----------------------------------

Install Python for the Mac ("MacPython") from python.org, or the
Enthought distribution if you want to build that version.

There is already a /usr/bin/python that at the moment is up to date, but
because this isn't always the case, and because the Python community seems to
strongly prefer it, we will base Visual on MacPython, which is /usr/local/bin/python.

Install wxPython, preferably the Cocoa version.

PYTHON INSTALLERS
--
The standard Python installers at python.org for the Mac install into `/Library/Frameworks/Python.framework/Versions/`.

For Python 2.x, the installer modified .profile to add the newly
installed Python to the application search PATH. The installer also
added or changed a link
`/Library/Frameworks/Python.framework/Versions/Current` to point to the
newly installed Python.

It will probably happen automatically but you should check:

    $ which python

to see that the correct python is being invoked from the command line. If not,
update your PATH environment variable (in .profile or equivalent) to make it so.

If you have several versions of python installed (yes... that would be
me). You'll want to make sure 'Current' is set correctly for whichever
version of python you're building against. I've found this script
useful:

     #!/bin/bash
     #
     # switchVersion.sh
     #
     cd /Library/Frameworks/Python.framework/Versions
     rm Current
     ln -s ./$1 Current
     ls -l Current
     echo "Now at version $1"

If you set this script to be executable you can just say
`$ ./switchVersion.sh 2.7` and it will switch the "Current" pointer
to version 2.7

Always do this before you build any of the dependencies or visual
python itself.

INSTALLING MODULES NEEDED BY VISUAL
----

You can get a numpy installer at http://www.numpy.org, or you can
build from source.  Enthought, naturally, comes with numpy
already. ;-)

setup.py will attempt to resolve other requirements at
https://pypi.python.org/pypi, but in case it's not successful you may
need to build from source ttfquery, polygon and fonttools.

Be sure to get ttfquery 1.0.4 or later.  Polygon is available at
http://polygon.origo.ethz.ch/download Currently, use the Polygon 2.0.1
binary for Python 2.6, which works with Python 27; For python 3, try
https://bitbucket.org/jraedler/polygon3/downloads

Note The following copyright notice applies to the Polygon module when
included in the VPython distribution concerning Polygon:

    "This distribution contains code from the GPC Library, and/or
    code resulting from the use of the GPC Library. This usage has been
    authorized by The University of Manchester, on the understanding that
    the GPC-related features are used only in the context of this
    distribution. It is not permitted to extract the GPC code from the
    distribution as the basis for commercial exploitation, unless a GPC
    Commercial Use Licence is obtained from The University of Manchester,
    contact: http://www.cs.man.ac.uk/~toby/gpc/".

BUILD VISUAL
---

You need Boost libraries for the Mac. See below for how to build the
Boost libraries.

Unpack the boost sources and cd into the boost_1.5.x directory.

Make sure "Current" is set, and for python3.2, or enthought, make sure
that /Developer/usr/bin is on your path.

For python 2.7, or enthought use this:

    ./bootstrap.sh --with-toolset=gcc --with-python-version=2.7 --with-python-root=/Library/Frameworks/Python.framework/Versions/Current --with-python=python2.7 --with-libraries=python,signals

# build for python3.2 use this

    ./bootstrap.sh --with-toolset=gcc --with-python-version=3.2 --with-python-root=/Library/Frameworks/Python.framework/Versions/Current --with-python=python3.2 --with-libraries=python,signals

Then for python.org python2.7 or python3.2 use this:

    ./b2 link=static threading=multi toolset=darwin cxxflags="-arch i386 -arch x86_64"

for enthought use this:

    ./b2 link=static threading=multi toolset=darwin cxxflags="-arch i386"

The result is in the 'stage' directory. I keep multiple versions of
the stage directory around to build various versions of visual, and
make a link to the one I'm using now:

    aluminum:boost_1_52_0 steve$ ls -lad stage*
    lrwxr-xr-x  1 steve  steve   15 Mar  2 09:51 stage -> stage-enthought
    drwxr-xr-x  3 steve  steve  102 Mar  1 09:38 stage-enthought
    drwxr-xr-x  3 steve  steve  102 Mar  1 11:06 stage-python.org-2.7
    drwxr-xr-x  3 steve  steve  102 Mar  1 11:17 stage-python.org-3.2

But if you are only worried about one version, you can just leave it
named "stage"

Finally, in your vpython-wx folder, make a subdirectory named
`dependencies` and a subdirectory to that named `boost_files`. In
there, add some links:

    ./dependencies//boost_files/boost -> ../../../boost_1_52_0/boost
    ./dependencies//boost_files/mac_libs -> ../../../boost_1_52_0/stage/lib

to match wherever you unpacked boost.

finally in the vpython-wx directory type:
    $ sudo python setup.py install

You should end up with a working visual python installation!

Something goes wrong, check above to see if you missed a step (I know.. there
are a lot of them!)

CREATE INSTALLER
-----

In src/mac/PackageMaker is an Apple PackageMaker project file, a
Welcome text file for creating a VPython installer for the Mac, and
instructions in packaging.txt.
