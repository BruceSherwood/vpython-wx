vpython-wx
==========

Experimental version of VPython (vpython.org) based on wxPython.
Current status (2012 Nov. 11) is that on Windows it can run all
of the standard VPython demo programs, but there are some details
that need to be worked on before a real release. Textures are not available on
Mac and Linux at the moment for lack of connection to the function
GetProcAddress(), which is platform-specific. The heart of the
GUI code is the file site-packages/visual_common/create_display.py.
It is imported by visual/__init__.py and by /vis/__init__.py; the
difference is that for convenience visual imports math and numpy.