"""
script to actually build the package. 

if the command line argument is 'pkg' then Uses the command line PackageMaker and the
pmdoc file to construct the package... 

if the command line argument is 'dmg' then use the hdutil program to create a .dmg

manual step to sign for now:

productsign --sign "Developer ID Installer: Silicon Prairie Ventures Inc" src/mac/PackageMaker/VisualPythonInstaller/VPython-Mac-Py2.7-6.11.pkg ~/outpath/VPython-Mac-6.02-Py2.7.pkg 

"""

import os
import sys
import subprocess

cwd = os.getcwd()
dir_part = os.path.split(cwd)[1]

if dir_part != 'vpython-wx':
    print("Please run this script from the vpython-wx directory, now in:" + os.getcwd())
    sys.exit(1)

import imp
setup = imp.load_source('setup',os.path.join(cwd,'setup.py'))
VERSION = setup.VERSION
HOME_DIR=os.path.expanduser("~")
PKG_MAKER_DIR="src/mac/PackageMaker"
PKG_MAKER="/Applications/PackageMaker.app/Contents/MacOS/PackageMaker"
PM_DOC=os.path.join(PKG_MAKER_DIR,'VPY-6-py27.pmdoc')
PKG_DIR=os.path.join(PKG_MAKER_DIR,'VisualPythonInstaller')
PKG_OUT=os.path.join(PKG_DIR,'VPython-Mac-Py2.7-' + VERSION + '.pkg')
DMG_OUT=os.path.join(PKG_MAKER_DIR,'VPython-Mac-Py2.7-' + VERSION + '.dmg')

if len(sys.argv)>1 and sys.argv[1]=='pkg':

    if not os.path.exists(PKG_DIR):
    	os.makedirs(PKG_DIR)

    if os.path.exists(PKG_OUT):
    	print "deleting old package"
    	os.remove(PKG_OUT)

    cmdList = [PKG_MAKER,'--verbose','--doc',PM_DOC,'--out',PKG_OUT]
    print "Executing:", cmdList
    subprocess.call(cmdList)

elif len(sys.argv)>1 and sys.argv[1]=='dmg':

    if os.path.exists(DMG_OUT):
        print "deleting old image"
    	os.remove(DMG_OUT)

    cmdList = ['hdiutil','create','-volname','VPython-Mac-Py2.7-' + VERSION,'-fs','HFS+','-srcfolder',PKG_DIR,DMG_OUT]
    print "Executing:", cmdList
    subprocess.call(cmdList)
else:
    print "Usage:", sys.argv[0], '[ pkg , dmg ]'
