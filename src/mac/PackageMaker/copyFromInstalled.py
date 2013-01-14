"""
This script copies a fully installed version of VPython (not FontTools, TTFQuery, Polygon, or wxPython)
from the site-packages directory to the staging area. These other packages change so rarely
that they only need to be copied very rarely.

You'll want to invoke this script with 'sudo' so that it runs with root privs.

"""
from __future__ import print_function

import subprocess
import os
import sys

#
# Edit this to match your staging directory etc.
#

STAGING_DIRECTORY = "../vpy-stage/site-packages/Visual Extension"
LIBRARY_DIRECTORY = "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages"

VPYTHON_EGG = "VPython-6.01-py2.7-macosx-10.6-intel.egg"
VPYTHON_PTH = "VPython.pth"

if os.getuid() != 0:
    print("You need to run this script as root")
    sys.exit(1)

if os.path.split(os.getcwd())[1] != 'vpython-wx':
    print("Please run this script from the vpython-wx directory, now in:" + os.getcwd())
    sys.exit(1)
    

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
