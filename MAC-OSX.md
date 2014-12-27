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


NOTES ABOUT XCODE5
------

I've been experimenting with various build/binary options for MacOSX/vpython during the last few weeks. I've managed to build VPython from the latest git using the tools that come with Xcode 5 (clang 5) so that it works with python.org python. The hardest part seems to be getting boost compiled/linked in a way that makes everybody happy. Here's what I have now for boost 1.55:

    ./bootstrap.sh --with-toolset=clang --with-python-version=2.7 --with-python-root=/Library/Frameworks/Python.framework/Versions/Current --with-python=python --with-libraries=python,signals

    ./b2 toolset=clang cxxflags="-stdlib=libstdc++ -arch i386 -arch x86_64" linkflags=-stdlib=libstdc++ link=static threading=multi

In the vpython source tree I've been setting:

aluminum:vpython-wx steve$ ls -laR dependencies/
total 0
drwxr-xr-x   3 steve  501   102 Feb 20  2013 .
drwxr-xr-x  48 steve  501  1632 Jun 17 15:30 ..
drwxr-xr-x   4 steve  501   136 Jun 16 06:47 boost_files

dependencies//boost_files:
total 16
drwxr-xr-x  4 steve  501  136 Jun 16 06:47 .
drwxr-xr-x  3 steve  501  102 Feb 20  2013 ..
lrwxr-xr-x  1 steve  501   27 Jun 16 06:46 boost -> ../../../boost_1_55_0/boost
lrwxr-xr-x  1 steve  501   31 Jun 16 06:47 mac_libs -> ../../../boost_1_55_0/stage/lib

Of course these can point to wherever your boost sources are living.

Once this is done I've been running:

    python setup.py build

in the vpython source directory, then:

    sudo python setup.py install

to install vpython in site-packages.

This works for python.org python and produces an "egg" distribution in site-packages:

"/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/VPython-6.05-py2.7-macosx-10.6-intel.egg"

My "quick and dirty" test is to run:

    python -c 'import visual; visual.sphere()'

from the command line to check it.

If you have ‘pip’ (<http://pip.readthedocs.org/en/latest/installing.html>) and 'virtualenv' installed (<http://virtualenv.readthedocs.org/en/latest/virtualenv.html>) you can create a virtual environment to play in and use wheels that I've created using this python/boost combo that works with scipy, ipython, etc. as follows:

    $ virtualenv ~/testvpy
    New python executable in testvpy/bin/python
    Installing setuptools, pip...done.

    $ source ~/testvpy/bin/activate
    (testvpy)aluminum:~ steve$ 

This is a "fresh" python environment with nothing installed

    (testvpy)aluminum:~ steve$ python
    Python 2.7.6 (v2.7.6:3a1db0d2747e, Nov 10 2013, 00:42:54) 
    [GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import numpy
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named numpy
    >>> import scipy
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named scipy
    >>> import visual
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ImportError: No module named visual

You can use "pip" to install stuff:

    (testvpy)aluminum:~ steve$ pip install numpy
    Downloading/unpacking numpy
      Downloading numpy-1.8.1-cp27-none-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.whl (12.0MB): 12.0MB downloaded
    Installing collected packages: numpy
    Successfully installed numpy
    Cleaning up...

    (testvpy)aluminum:~ steve$ pip install scipy
    Downloading/unpacking scipy
      Downloading scipy-0.14.0-cp27-none-macosx_10_6_intel.macosx_10_9_intel.macosx_10_9_x86_64.whl (26.7MB): 26.7MB downloaded
    Installing collected packages: scipy
    Successfully installed scipy
    Cleaning up...

I've put some wheels up to enable vpython  (built as above) to work. If you are willing, I'd appreciate if you'd try this:

    (testvpy)aluminum:~ steve$ pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/VPython-6.05-cp27-none-macosx_10_6_intel.whl
    Downloading/unpacking https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/VPython-6.05-cp27-none-macosx_10_6_intel.whl
      Downloading VPython-6.05-cp27-none-macosx_10_6_intel.whl (8.7MB): 8.7MB downloaded
    Installing collected packages: VPython
    Successfully installed VPython
    Cleaning up...

    (testvpy)aluminum:~ steve$ pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/TTFQuery-1.0.5-py2-none-any.whl
    Downloading/unpacking https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/TTFQuery-1.0.5-py2-none-any.whl
      Downloading TTFQuery-1.0.5-py2-none-any.whl
    Installing collected packages: TTFQuery
    Successfully installed TTFQuery
    Cleaning up...

    (testvpy)aluminum:~ steve$ pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/FontTools-2.4-cp27-none-macosx_10_6_intel.whl
    Downloading/unpacking https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/FontTools-2.4-cp27-none-macosx_10_6_intel.whl
      Downloading FontTools-2.4-cp27-none-macosx_10_6_intel.whl (342kB): 342kB downloaded
    Requirement already satisfied (use --upgrade to upgrade): numpy in ./testvpy/lib/python2.7/site-packages (from FontTools==2.4)
    Installing collected packages: FontTools
    Successfully installed FontTools
    Cleaning up...

    (testvpy)aluminum:~ steve$ pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/Polygon2-2.0.6-cp27-none-macosx_10_6_intel.whl
    Downloading/unpacking https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/Polygon2-2.0.6-cp27-none-macosx_10_6_intel.whl
      Downloading Polygon2-2.0.6-cp27-none-macosx_10_6_intel.whl (77kB): 77kB downloaded
    Installing collected packages: Polygon2
    Successfully installed Polygon2
    Cleaning up...

There is no "pip" way to install wxpython yet, so I just linked the wxredirect.pth file from my python.org installation of wxPython:

    ln -s /Library/Frameworks/Python.framework/Versions/Current/lib/python2.7/site-packages/wxredirect.pth ~/testvpy/lib/python2.7/site-packages/
    
Also need a link to the pythonw binary:

    ln -s /Library/Frameworks/Python.framework/Versions/Current/bin/pythonw ~/testvpy/bin
    
Finally add a PYTHONHOME definition in the activate script:

    printf '\nPYTHONHOME="$VIRTUAL_ENV"\n' >> ~/testvpy/bin/activate

Now, after you "reactivate" (close the old terminal, start a fresh one, and type:):

    $ source ~/testvpy/bin/activate

You can do a quick test:

    pythonw -c 'import visual; visual.sphere()'

--------------------------------------

Notes for installing vpython on MacOSX for Enthought Canopy:

You can install vpython in the main canopy environment by running the canopy application and selecting:

Tools -> Canopy Terminal

You can install from wheels by upgrading pip. Download:

<https://bootstrap.pypa.io/get-pip.py>

and execute:

(Canopy 64bit) $python ~/Downloads/get-pip.py 

Then install from binary wheels on Dropbox:

(Canopy 64bit) $pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/canopy/VPython-6.10-cp27-none-macosx_10_6_x86_64.whl
(Canopy 64bit) $pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/canopy/Polygon2-2.0.6-cp27-none-macosx_10_6_x86_64.whl
(Canopy 64bit) $pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/canopy/FontTools-2.4-cp27-none-macosx_10_6_x86_64.whl
(Canopy 64bit) $pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/canopy/TTFQuery-1.0.4-py2-none-any.whl

*OR*

You can install vpython in a virtual environment without disturbing your global canopy environment

You can install from wheels by upgrading pip. Download:

<https://bootstrap.pypa.io/get-pip.py>

and execute:

~/my_venv/bin/python ~/Downloads/get-pip.py 

Then install from binary wheels on Dropbox:

~/my_venv/bin/pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/canopy/VPython-6.10-cp27-none-macosx_10_6_x86_64.whl
~/my_venv/bin/pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/canopy/Polygon2-2.0.6-cp27-none-macosx_10_6_x86_64.whl
~/my_venv/bin/pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/canopy/FontTools-2.4-cp27-none-macosx_10_6_x86_64.whl
~/my_venv/bin/pip install https://dl.dropboxusercontent.com/u/20562746/VPythonWheels/canopy/TTFQuery-1.0.4-py2-none-any.whl

*OR*

You can build vpython from source, but you'll need to built boost as well:

in the boost boost_1_55_0 directory execute:

./bootstrap.sh --with-toolset=clang --with-python-version=2.7 --with-python=python --with-python-root=/Applications/Canopy.app/appdata/canopy-1.4.0.1938.macosx-x86_64/Canopy.app/Contents/ --with-libraries=python,signals

./b2 toolset=clang cxxflags="-stdlib=libstdc++  -arch x86_64" linkflags=-stdlib=libstdc++ -j2  link=static threading=multi

then in the vpython-wx source directory follow the same instructions from above for XCODE5.

-------------------------------

Notes for installing vpython on MacOSX for brew:

brew install boost --c++11 --with-python

Then build per above instructions for XCODE5. 

You don't need to worry about the 'dependencies' folder since brew puts boost libs in /usr/local/lib where the compiler will find them by default.

Brew also has a formula for wxpython which is new enough (3.0+) to work with visual python 6 if you want to use that one.
