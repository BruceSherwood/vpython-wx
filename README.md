vpython-wx
==========
[![Build Status](https://travis-ci.org/BruceSherwood/vpython-wx.png?branch=master,stable)](https://travis-ci.org/BruceSherwood/vpython-wx)

Experimental version of VPython (vpython.org) based on the cross-platform library wxPython. It differs from the older VPython (5.74 and before) by eliminating nearly all platform-dependent code and by eliminating the threading associated with rendering. 

The new version makes one essential change to the syntax of VPython programs. Now, an animation loop MUST contain a rate statement, which limits the number of loop iterations per second as before but also when appropriate (about 30 times per second) updates the 3D scene and handles mouse and keyboard events. Without a rate statement, the scene will not be updated until and unless the loop is completed.

The heart of the user-interface code (creating windows and handling events) is the file site-packages/visual_common/create_display.py. It is imported by visual/__init__.py and by /vis/__init__.py; the difference is that for convenience visual imports math and numpy, whereas vis doesn't.

Please report issues to the [Github repository](https://github.com/BruceSherwood/vpython-wx), or to the VPython mailing list.
