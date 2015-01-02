Mac Visual Python Installer
===========================

Please find enclused the Visual Python .pkg installer and a very simple
uninstall script. The intro to the .pkg installer reads:

--------------------------------------------------------------------------------

The Python.org “Python 2.7” distribution must first be installed from python.org 
before installing this Python module. This installer installs everything you need 
(except for Python itself) to edit and run vpython programs on MacOS X. In 
addition to the visual module (6.11), this installer installs numpy-1.9.1, 
wxPython-3.0.2, TTFQuery-1.05, Polygon-2.07 and FontTools-2.4.

--------------------------------------------------------------------------------

If you've used an earlier version of the VPython-6 installer and wish to
uninstall before installing the new version (not strictly necessary, but
tidier) you can use the included script:

$ sudo sh -v vpy_uninstall.sh

Or you can try the "vpy_uninstall.app" which is a simple Automator
wrapper around the shell script. Note that "vpy_uninstall.app" will
only work if the .dmg is mounted since it simply calls the script
that's located on the .dmg!
