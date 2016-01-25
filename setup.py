from __future__ import print_function

import sys
import os
from glob import glob
from setuptools import Extension, setup
import numpy
import platform

# This file must be placed at the top level of the GitHub project directory
# and then be executed from a terminal as "python setup.py build"

VERSION = '6.11'
DESCRIPTION = '3D Programming for Ordinary Mortals'
LONG_DESCRIPTION = """
VPython is the Python programming language plus a 3D graphics module
called "Visual" originated by David Scherer in 2000. VPython makes it
easy to create navigable 3D displays and animations, even for those
with limited programming experience. Because it is based on Python, it
also has much to offer for experienced programmers and researchers."""

classifiers = filter(None, [x.strip() for x in """
Intended Audience :: Education
Intended Audience :: Developers
Intended Audience :: Science/Research
Development Status :: 4 - Beta
Environment :: MacOS X :: Cocoa
Environment :: Win32 (MS Windows)
Environment :: X11 Applications :: GTK
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: POSIX :: BSD
Programming Language :: Python :: 2.7
Topic :: Multimedia :: Graphics :: 3D Modeling
Topic :: Scientific/Engineering :: Physics
Topic :: Scientific/Engineering :: Visualization
Topic :: Software Development :: Libraries :: Python Modules
""".split('\n')])

VISUAL_DIR = os.getcwd()
VISUAL_INC = os.path.join(VISUAL_DIR,'include')

if 'build' in sys.argv:
    try:
        import wx
    except ImportError:
        raise RuntimeError("Sorry.. you need to install wxPython (at least 2.9.x) to build this version of vpython")

    if tuple([int(x) for x in wx.__version__.split('.')]) < (2,9):
        raise RuntimeError("Sorry.. you need to install wxPython (at least 2.9.x) to build this version of vpython")

# In python 2.6, version_info is just a tuple.  In 2.7, it is a namedTuple.
# This code should work in 2.6 and 2.7
major = sys.version_info[0]
minor = sys.version_info[1]
versionString = ''.join([str(major), str(minor)])

os_host = platform.platform(terse=True).split('-')[0].lower()
if os_host=='darwin':
    os_host = 'mac'

if os_host=='windows':
    BOOST_DIR = os.path.join(VISUAL_DIR,os.path.join('dependencies','boost_files'))
    BOOST_LIBDIR = os.path.join(BOOST_DIR,'windows_libs')
    LIBRARY_DIRS = [BOOST_LIBDIR]

elif os_host in ('linux'):
    from get_vpy_includes import get_includes, get_libs

    LINK_FLAGS="-Wl,--export-dynamic"

    GTK_VIS_LIBS = get_libs()
    # GTK_VIS_LIBS.append('boost_python-mt-py' + versionString)
    GTK_VIS_LIBS.append('boost_python-py' + versionString)
    GTK_VIS_LIBS.append('boost_signals')

    GTK_INCDIRS = get_includes()

    LIBRARY_DIRS=[]

elif os_host in ('freebsd'):
    from get_vpy_includes import get_includes, get_libs

    LINK_FLAGS="-Wl,--export-dynamic"

    GTK_VIS_LIBS = get_libs()
    # freebsd ports install libboost_python for py27
    GTK_VIS_LIBS.append('boost_python')
    GTK_VIS_LIBS.append('boost_signals')

    GTK_INCDIRS = get_includes()

    LIBRARY_DIRS=[]


elif os_host == 'mac':
    if not os.path.exists('setup.cfg'):
        #
        # we have no setup.cfg file, assume boost is in 'dependencies'
        #
        BOOST_DIR = os.path.join(VISUAL_DIR,os.path.join('dependencies','boost_files'))
        BOOST_LIBDIR = os.path.join(BOOST_DIR,'mac_libs')
        LIBRARY_DIRS = [BOOST_LIBDIR]
    else:
        BOOST_LIBDIR=[] # let setup.cfg handle it
        LIBRARY_DIRS=[]

else:
    BOOST_LIBDIR=[]
    LIBRARY_DIRS=[]

INCLUDE_DIRS = [
    numpy.get_include(),
    VISUAL_INC,
    os.path.join(VISUAL_INC,'util'),
    os.path.join(VISUAL_INC,'python'),
    ]

if os_host in ('windows','mac'):
    if os_host == 'mac':
        INCLUDE_DIRS.append(os.path.join(VISUAL_INC,'mac'))
        if not os.path.exists('setup.cfg'):
            INCLUDE_DIRS.append(BOOST_DIR)
    else:
        INCLUDE_DIRS.append(BOOST_DIR)
        INCLUDE_DIRS.append(os.path.join(VISUAL_INC,'win32'))

elif os_host == 'linux':
    INCLUDE_DIRS.append(os.path.join(VISUAL_INC,'gtk2'))
    INCLUDE_DIRS += GTK_INCDIRS

elif os_host == 'freebsd':
    INCLUDE_DIRS.append(os.path.join(VISUAL_INC,'gtk2'))
    INCLUDE_DIRS += GTK_INCDIRS

VISUAL_SOURCES = []

patterns = ["/src/core/*.cpp","/src/core/util/*.cpp","/src/python/*.cpp",]

if os_host == 'mac':
    patterns.append("/src/mac/*.cpp")
elif os_host == 'linux':
    patterns.append("/src/gtk2/*.cpp")
elif os_host == 'freebsd':
    patterns.append("/src/gtk2/*.cpp")
elif os_host == 'windows':
    patterns.append("/src/win32/*.cpp")

for pattern in patterns:
    VISUAL_SOURCES.extend(glob(VISUAL_DIR + pattern))

if os_host == 'mac':
    os.environ['LDFLAGS'] = '-framework Cocoa -framework OpenGL' # -framework Python'
elif os_host == 'linux':
    os.environ['LD_FLAGS'] = LINK_FLAGS

extra_compile_args=[]

if os_host == 'mac':
    libraries = ['boost_python','boost_signals']
    extra_compile_args.append('-F/System/Library/Frameworks/OpenGL.framework')
elif os_host == 'windows':
    libraries=['opengl32', 'glu32', 'user32', 'advapi32', 'gdi32']
    extra_compile_args.append('/DM_PI=3.1415926535897932384626433832795')
elif os_host == 'linux':
    libraries=GTK_VIS_LIBS
elif os_host == 'freebsd':
    libraries=GTK_VIS_LIBS

CVISUAL = Extension(
    "visual_common.cvisual",
    VISUAL_SOURCES,
    include_dirs=INCLUDE_DIRS,
    library_dirs=LIBRARY_DIRS,
    libraries=libraries,
    extra_compile_args = extra_compile_args,
    )

SITE_PACKAGES = os.path.join(VISUAL_DIR, 'site-packages')
VIDLE_PATH = os.path.join(SITE_PACKAGES,'vidle%i' % sys.version_info[0])

def main():

    setup(
        name='VPython',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        classifiers=classifiers,
        author='David Scherer et al.',
        author_email='visualpython-users@lists.sourceforge.net',
        platforms=['POSIX','MacOS','Windows'],
        license='other',
        url='http://www.vpython.org/',
        version=VERSION,
        packages=['visual', 'vis', 'vidle', 'visual_common'],
        package_dir={
            'visual': os.path.join(SITE_PACKAGES,'visual'),
            'vis': os.path.join(SITE_PACKAGES,'vis'),
            'visual_common':os.path.join(SITE_PACKAGES,'visual_common'),
            'vidle': VIDLE_PATH,
            },
        package_data={'vidle':['*.txt', '*.def','Icons/*.icns'],
                      'visual':['*.txt', 'docs/*.html', 'docs/*.gif', 'docs/*.pdf', 'docs/*.js',
                                'docs/images/*.jpg', 'docs/*.txt', 'docs/*.css',
                                'examples/*.py', 'examples/*.tga'],
                      'visual_common':['*.tga'],
                      },
        ext_modules=[CVISUAL],
        requires=[
            'fontTools',
            'numpy',
            'Polygon',
            'ttfquery',
            'wx',
            #,'wxPython >= 2.9'],
        ],
        zip_safe=False)


if __name__=='__main__':
    main()
