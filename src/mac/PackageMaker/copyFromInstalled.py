"""
This script copies a fully installed version of VPython (not FontTools, numpy, TTFQuery, Polygon, or wxPython)
from the site-packages directory to the staging area. These other packages change so rarely
that they only need to be copied very rarely.

You'll want to invoke this script with 'sudo' so that it runs with root privs.

Execute this from the 'vpython-wx' directory, which should be the peer of the vpy-stage directory

"""
from __future__ import print_function

import subprocess
import os
import sys

if os.getuid() != 0:
    print("You need to run this script as root")
    sys.exit(1)


cwd = os.getcwd()
dir_part = os.path.split(cwd)[1]

if dir_part != 'vpython-wx':
    print("Please run this script from the vpython-wx directory, now in:" + os.getcwd())
    sys.exit(1)

import imp
print("Loading from:", os.path.join(cwd,'setup.py'))
setup = imp.load_source('setup',os.path.join(cwd,'setup.py'))
VPYTHON_WX_VERSION = setup.VERSION
    
#
# Edit this to match your staging directory etc.
#

VPYTHON_WX_DIR=os.getcwd()
STAGING_DIRECTORY = os.path.join(os.path.split(VPYTHON_WX_DIR)[0],"vpy-stage/site-packages/Visual Extension")
LIBRARY_DIRECTORY = "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages"

VPYTHON_EGG = "VPython-" + VPYTHON_WX_VERSION + "-py2.7-macosx-10.6-intel.egg"
VPYTHON_PTH = "VPython.pth"

src = os.path.join(LIBRARY_DIRECTORY, VPYTHON_EGG)

cpyList = ['cp','-R',os.path.join(LIBRARY_DIRECTORY, VPYTHON_EGG), os.path.join(STAGING_DIRECTORY,'.')]

if subprocess.call(cpyList):
    raise RuntimeError("Sorry... copy didn't work.")
else:
    print("Egg copied")
    
cpyList = ['cp','-R',os.path.join(LIBRARY_DIRECTORY, VPYTHON_PTH), os.path.join(STAGING_DIRECTORY,'.')]

if subprocess.call(cpyList):
    raise RuntimeError("Sorry... copy didn't work.")
else:
    print("Path file copied")

sedList = ['sed','-e','s/Classic\ Windows/Classic\ OSX/g','-i','.orig',os.path.join(STAGING_DIRECTORY, VPYTHON_EGG, 'vidle','config-main.def')]
if subprocess.call(sedList):
    raise RuntimeError("Sorry... key config substitution didn't work.")
else:
    print("Key Config changed")
#
# Everything is copied... now clear out .pyc files from examples and force
# permissions and ownership
# 

exampDir = os.path.join(STAGING_DIRECTORY,VPYTHON_EGG,'visual','examples')
os.chdir(exampDir)

deleted = []
for fname in os.listdir('.'):
    fpath = os.path.join(exampDir, fname)
    if fname.endswith('pyc'):
        try:
            os.unlink(fname)
            deleted.append(fname)
        except:
            print("removing:", fpath, "failed")

    if fname.startswith('__init__') and fname not in deleted:
        try:
            os.unlink(fname)
        except:
            print("removing ", fpath, "failed")

print("Examples pyc files cleared out")

os.chdir('../../..')

if subprocess.call(['chown','-R','root:admin','.']):
    raise RuntimeError("Sorry... chown didn't work.")

print("chown successful")

if subprocess.call(['chmod','-R','g+w','.']):
    raise RuntimeError("Sorry... chmod didn't work.")

print("chmod successful")

if subprocess.call(['find','.','-name','.DS_Store','-exec','rm','{}',';','-print']):
    raise RuntimeError("Sorry... .DS_Store clearout didn't work.")
    
print("Cleared out .DS_Store files")
