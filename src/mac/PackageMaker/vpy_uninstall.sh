#!/bin/bash
#
# This is an experimental uninstall tool for VPython. It will uninstall VPython 
# and its dependencies from the python.org python site packages and wxPython
# from /usr/local/lib
#

SITE_PATH="/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages"
LIB_PATH="/usr/local/lib"
APP_PATH="/Applications"

for pck in 'FontTools' 'TTFQuery' 'ttfquery' 'Polygon' 'numpy' 'wxredirect' 'VPython'
do
    rm -r $SITE_PATH/$pck*
done

rm -r $LIB_PATH/wxPython*
rm -r $APP_PATH/VIDLE-Py2.7.app
