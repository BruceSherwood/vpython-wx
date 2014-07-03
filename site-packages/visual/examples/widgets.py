from __future__ import division, print_function
from visual import *
from visual.graph import *
import wx

# wx is the wxPython library (see wxpython.org)

# wxPython is a Python library that makes it possible to create
# windows and handle events cross-platform, with native look-and-feel on
# Windows, Mac, and Linux. This program widgets.py uses VPython to handle
# 3D graphics, and wxPython statements to create buttons, sliders, etc.

# Among the documentation mentioned at wxpython.org, a particularly
# useful reference manual is docs.wxwidgets.org/3.0. At that site, the
# section "Categories" gives a useful overview of the components of wxPython,
# and the category "Controls" describes all the things you can do with
# buttons, sliders, etc.

# When you see "wxButton" that actually means wx.Button, after importing
# wx. The wxPython library is based on wxWidgets, a library for C++ programs,
# and the documentation for wxPython and wxWidgets is very similar.

# Because most wxPython classes inherit attributes and methods from
# other, more general wxPython classes, investigate the parent classes to
# learn about capabilities that may not be mentioned explicitly in the
# inheriting class.

# For simplicity, this program places various widgets at specific
# positions within the window. However, wxPython also offers the
# option to allow it to rearrange the positioning of widgets as
# a function of window size and shape. A good tutorial on wxPython,
# which includes a discussion of "Layout Management", is found at
# zetcode.com/wxpython.

# Functions that are called on various events

def setleft(evt): # called on "Rotate left" button event
    cube.dir = -1

def setright(evt): # called on "Rotate right" button event
    cube.dir = 1

def sethide(evt): # called on "Hide for 3 s" button event
    w.visible = False
    sleep(3)
    w.visible = True

def setfull(evt): # called on "Full screen for 3 s" button event
    w.fullscreen = True
    sleep(3)
    w.fullscreen = False

def leave(evt): # called on "Exit under program control" button event
    exit()

def setred(evt): # called by "Make red" menu item
    cube.color = color.red
    t1.SetSelection(0) # set the top radio box button (red)

def setcyan(evt): # called by "Make cyan" menu item
    cube.color = color.cyan
    t1.SetSelection(1) # set the bottom radio box button (cyan)

def togglecubecolor(evt): # called by radio box (a set of two radio buttons)
    choice = t1.GetSelection()
    if choice == 0: # upper radio button (choice = 0)
        cube.color = color.red
    else: # lower radio button (choice = 1)
        cube.color = color.cyan

def cuberate(value):
    cube.dtheta = 2*value*pi/1e4
    
def setrate(evt): # called on slider events
    value = s1.GetValue()
    cuberate(value) # value is min-max slider position, 0 to 100

L = 320
Hgraph = 400
# Create a window. Note that w.win is the wxPython "Frame" (the window).
# window.dwidth and window.dheight are the extra width and height of the window
# compared to the display region inside the window. If there is a menu bar,
# there is an additional height taken up, of amount window.menuheight.
# The default style is wx.DEFAULT_FRAME_STYLE; the style specified here
# does not enable resizing, minimizing, or full-sreening of the window.
w = window(width=2*(L+window.dwidth), height=L+window.dheight+window.menuheight+Hgraph,
           menus=True, title='Widgets',
           style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)

# Place a 3D display widget in the left half of the window.
d = 20
disp = display(window=w, x=d, y=d, width=L-2*d, height=L-2*d, forward=-vector(0,1,2))
gdisplay(window=w, y=disp.height+50, width=2*(L+window.dwidth), height=Hgraph)

cube = box(color=color.red)

# Place buttons, radio buttons, a scrolling text object, and a slider
# in the right half of the window. Positions and sizes are given in
# terms of pixels, and pos(0,0) is the upper left corner of the window.
p = w.panel # Refers to the full region of the window in which to place widgets

wx.StaticText(p, pos=(d,4), size=(L-2*d,d), label='A 3D canvas',
              style=wx.ALIGN_CENTRE | wx.ST_NO_AUTORESIZE)

left = wx.Button(p, label='Rotate left', pos=(L+10,15))
left.Bind(wx.EVT_BUTTON, setleft)

right = wx.Button(p, label='Rotate right', pos=(1.5*L+10,15))
right.Bind(wx.EVT_BUTTON, setright)

hide = wx.Button(p, label='Hide for 3 s', pos=(L+10,50))
hide.Bind(wx.EVT_BUTTON, sethide)

full = wx.Button(p, label='Full screen for 3 s', pos=(1.5*L+10,50))
full.Bind(wx.EVT_BUTTON, setfull)

exit_program = wx.Button(p, label='Exit under program control', pos=(L+70,200))
exit_program.Bind(wx.EVT_BUTTON, leave)

t1 = wx.RadioBox(p, pos=(1.0*L,0.3*L), size=(0.25*L, 0.25*L),
                 choices = ['Red', 'Cyan'], style=wx.RA_SPECIFY_ROWS)
t1.Bind(wx.EVT_RADIOBOX, togglecubecolor)

# On the Mac, wx.TextCtrl is resized when the window is resized.
# This resizing does not occur on Windows or Linux. Unlike for
# wx.StaticText used above, there is no wx.ST_NO_AUTORESIZE option.
tc = wx.TextCtrl(p, pos=(1.4*L,90), value='You can type here:\n',
            size=(150,90), style=wx.TE_MULTILINE)
tc.SetInsertionPoint(len(tc.GetValue())+1) # position cursor at end of text
tc.SetFocus() # so that keypresses go to the TextCtrl without clicking it
# Note that disp.canvas.SetFocus() will put disp in keyboard focus.

s1 = wx.Slider(p, pos=(1.0*L,0.8*L), size=(0.9*L,20), minValue=0, maxValue=100)
s1.Bind(wx.EVT_SCROLL, setrate)
wx.StaticText(p, pos=(1.0*L,0.75*L), label='Set rotation rate')

# Create a menu of options (Rotate right, Rotate right, Make red, Make cyan).
# Currently, menus do not work on the Macintosh.
m = w.menubar # Refers to the menubar, which can have several menus

menu = wx.Menu()
item = menu.Append(-1, 'Rotate left', 'Make box rotate to the left')
w.win.Bind(wx.EVT_MENU, setleft, item)

item = menu.Append(-1, 'Rotate right', 'Make box rotate to the right')
w.win.Bind(wx.EVT_MENU, setright, item)

item = menu.Append(-1, 'Make red', 'Make box red')
w.win.Bind(wx.EVT_MENU, setred, item)

item = menu.Append(-1, 'Make cyan', 'Make box cyan')
w.win.Bind(wx.EVT_MENU, setcyan, item)

# Add this menu to an Options menu next to the default File menu in the menubar
m.Append(menu, 'Options')

# Initializations
s1.SetValue(70) # update the slider
cuberate(s1.GetValue()) # set the rotation rate of the cube
cube.dir = -1 # set the rotation direction of the cube

# Add a graph to the window
funct1 = gcurve(color=color.cyan)
funct2 = gvbars(delta=0.5, color=color.red)
funct3 = gdots(color=color.yellow)

for t in arange(-30, 74, 1):
    funct1.plot( pos=(t, 5.0+5.0*cos(-0.2*t)*exp(0.015*t)) )
    funct2.plot( pos=(t, 2.0+5.0*cos(-0.1*t)*exp(0.015*t)) )
    funct3.plot( pos=(t, 5.0*cos(-0.03*t)*exp(0.015*t)) )

# A VPython program that uses these wxPython capabilities should always end
# with an infinite loop containing a rate statement, as future developments
# may require this to keep a display active. It can be as simple as
# while True: rate(1)
while True:
    rate(100)
    cube.rotate(axis=(0,1,0), angle=cube.dir*cube.dtheta)
       
