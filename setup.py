import sys 
import os 
from glob import glob 
from setuptools import Extension, setup 
import numpy 
import platform

# This file must be placed at the top level of the GitHub project directory
# and then be executed from a terminal as "python setup.py build"
 
VERSION = '6.01' 
DESCRIPTION = '3D Programming for Ordinary Mortals' 
LONG_DESCRIPTION = """ 
VPython is the Python programming language plus a 3D graphics module 
called "Visual" originated by David Scherer in 2000. VPython makes it 
easy to create navigable 3D displays and animations, even for those 
with limited programming experience. Because it is based on Python, it 
also has much to offer for experienced programmers and researchers.""" 
 
classifiers = """
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
""".split()
 
VISUAL_DIR = os.getcwd() 
VISUAL_INC = os.path.join(VISUAL_DIR,'include') 
 
if 'build' in sys.argv: 
    try: 
        import wx 
    except ImportError: 
        raise RuntimeError, "Sorry.. you need to install wxPython (at least 2.9.x) to build this version of vpython" 
 
    if tuple([int(x) for x in wx.__version__.split('.')]) < (2,9): 
        raise RuntimeError, "Sorry.. you need to install wxPython (at least 2.9.x) to build this version of vpython" 
     
versionString = ''.join([`sys.version_info.major`, `sys.version_info.minor`]) 
 
os_host = platform.platform(terse=True).split('-')[0].lower() 
if os_host=='darwin': 
    os_host = 'mac' 
 
if os_host in ('windows','mac'): 
    BOOST_DIR = os.path.join(VISUAL_DIR,os.path.join('dependencies','boost_files')) 
 
if os_host=='mac': 
    BOOST_LIBDIR = os.path.join(BOOST_DIR,'mac_libs') 
elif os_host=='windows': 
    BOOST_LIBDIR = os.path.join(BOOST_DIR,'windows_libs') 
     
 
if os_host in ('windows','mac'): 
    LIBRARY_DIRS = [BOOST_LIBDIR] 
 
elif os_host in ('linux'): 
    from get_vpy_includes import get_includes, get_libs 
 
    LINK_FLAGS="-Wl,--export-dynamic" 
     
    GTK_VIS_LIBS = get_libs()
    GTK_VIS_LIBS.append('boost_python-mt-py' + versionString) 
    GTK_VIS_LIBS.append('boost_signals') 
     
    GTK_INCDIRS = get_includes() 
 
    LIBRARY_DIRS=[] 
     
     
INCLUDE_DIRS = [ 
    numpy.get_include(), 
    VISUAL_INC, 
    os.path.join(VISUAL_INC,'util'), 
    os.path.join(VISUAL_INC,'python'), 
    ] 
     
if os_host in ('mac','windows'): 
    INCLUDE_DIRS.append(BOOST_DIR) 
    if os_host == 'mac': 
        INCLUDE_DIRS.append(os.path.join(VISUAL_INC,'mac')) 
    else: 
        INCLUDE_DIRS.append(os.path.join(VISUAL_INC,'win32')) 
         
elif os_host == 'linux': 
    INCLUDE_DIRS.append(os.path.join(VISUAL_INC,'gtk2')) 
    INCLUDE_DIRS += GTK_INCDIRS 
 
VISUAL_SOURCES = [] 
 
patterns = ["/src/core/*.cpp","/src/core/util/*.cpp","/src/python/*.cpp",] 
 
if os_host == 'mac': 
    patterns.append("/src/mac/*.cpp") 
elif os_host == 'linux': 
    patterns.append("/src/gtk2/*.cpp") 
elif os_host == 'windows': 
    patterns.append("/src/win32/*.cpp") 
 
for pattern in patterns: 
    VISUAL_SOURCES.extend(glob(VISUAL_DIR + pattern)) 
 
if os_host == 'mac': 
    os.environ['LDFLAGS'] = '-framework Cocoa -framework OpenGL -framework Python' 
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

setup( 
    name='VPython', 
    description=DESCRIPTION, 
    long_description=LONG_DESCRIPTION,
    classifiers=classifiers,
    author='David Scherer et al.',
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
                  'visual':['*.txt','docs/*.html', 'docs/visual/*.html', 'docs/visual/*.gif', 'docs/visual/*.pdf',
                            'docs/visual/images/*.jpg', 'docs/visual/*.txt', 'docs/visual/*.css', 'examples/*.py'],
                  'visual_common':['*.tga'],
                  },
    ext_modules=[CVISUAL],
    install_requires=['Polygon >= 2.0, <3.0', 'FontTools >= 2.0', 'TTFQuery >= 1.0','wxPython >= 2.9'], # <- there is no wxPython 2.9 on PyPy
    zip_safe=False) 
