from __future__ import print_function

# Note that Python variables starting with "_" are not accessible to the importing program,
# so variables intended to be private in this module start with "_" 

# When running from a terminal get this warning:
# swig/python detected a memory leak of type wxPlatformInfo *', no destructor found.
# Probably doesn't matter.

import wx as _wx
try:
    from wx import glcanvas
except ImportError:
    print("\n\nUnable to import glcanvas\n\n")
import os as _os
import sys as _sys
import time as _time
import numpy as _numpy
import atexit as _atexit
from inspect import getargspec

version = ('6.11', 'release')

import visual_common.cvisual as cvisual

from visual_common.cvisual import (vector, mag, mag2, norm, cross, rotate,
                       comp, proj, diff_angle, set_wait)
from visual_common.primitives import (arrow, cylinder, cone, sphere, box, ring, label,
                               frame, pyramid, ellipsoid, curve, faces, convex, helix,
                               points, text, distant_light, local_light, extrusion,
                               text_to_bitmap, trail_list)

import visual_common.crayola as crayola
color = crayola
import visual_common.materials as materials
import visual_common.shapes as shapes
import visual_common.paths as paths

try:
    from Polygon import Polygon
except:
    pass

def wait(*args): # called by mouseobject.cpp/pop_click, which is called by scene.mouse.getclick()
    _Interact()
    if len(args) == 0: return
    elif len(args) == 1: _time.sleep(args[0])
    else: raise ValueError("Too many arguments for the wait() function.")

_App = _wx.App()

_platInfo = _wx.PlatformInformation()
_plat = _platInfo.GetOperatingSystemFamilyName() # 'Windows', 'Macintosh'

_screenwidth, _screenheight = _wx.GetDisplaySize()

def exit():
    _wx.Exit()

class window(object):
    
    @classmethod
    def delete_all(cls):
        ds = _displays.displays[:] # make a copy
        for d in ds:
            try:
                d.window.delete()
            except:
                pass
        
    def __init__(self, menus=False, _make_panel=True, x=0, y=0, width=600, height=600,
                 title='VPython',visible=True, fullscreen=False,
                 style=_wx.DEFAULT_FRAME_STYLE, **keywords):
        global _do_loop
        _do_loop = True # if a window has been made, execute wait loop at exit
        self.menus = menus
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        if self.x > _screenwidth - 100:
            raise ValueError('window x too near the right edge of the screen')
        if self.x + self.width < 100:
            raise ValueError('window x too far to the left edge of the screen')
        if self.y > _screenheight - 100:
            raise ValueError('window y too close to the bottom of the screen')
        if self.y + self.height < 100:
            raise ValueError('window y too far to the top of the screen')

        if self.x + self.width > _screenwidth:
            self.width = _screenwidth - self.x
        if self.x < 0:
            self.width = self.x + self.width
            self.x = 0
        if self.y + self.height > _screenheight:
            self.height = _screenheight - self.y
        if self.y < 0:
            self.height = self.y + self.height
            self.y = 0
            
        if _plat == 'Macintosh': self.y += 20 # 20-pixel menu bar at top of screen
        if _plat == 'Unix':
            if (self.x <= 65): self.x = 0 # 65-pixel band (launcher) at left of screen for Ubuntu 12.04
            self.y += 25 # 25-pixel band (menus etc.) at top of screen for Ubuntu 12.04
##        if _plat == 'Unix': self.height -= 25 # correction for Xubuntu; not understood
        self.win = _wx.Frame(None, -1, title, pos=(self.x, self.y),
                             size=(self.width, self.height), style=style)
        _displays.window_num += 1
        self.N = _displays.window_num
        self._canvas = None # not None if one canvas fills the window
        self.menubar = None

        self._visible = visible
        self._fullscreen = fullscreen
        self._fullscreen_count = 0
        self._exit = True
        if 'exit' in keywords:
            self._exit = keywords['exit']
            del keywords['exit']
        for kw in keywords:
            setattr(self, kw, keywords[kw])
        self.cursor = cursor(win=self.win)
            
        self.win.Bind(_wx.EVT_CLOSE, self._OnExitApp)
        self.win.Bind(_wx.EVT_MOVE, self._OnMove)
        self.win.Bind(_wx.EVT_SIZE, self._OnSize)
        self.win.Bind(_wx.EVT_MAXIMIZE, self._OnMaximize)
        self.win.Bind(_wx.EVT_ICONIZE, self._OnIconize)
        
        if _make_panel: self.panel = _wx.Panel(self.win)
        else: self.panel = None

        if self.menus:
            self.menubar = _wx.MenuBar()
            menu = _wx.Menu()
            item = menu.Append(_wx.ID_EXIT, "E&xit\tCtrl-Q", "Exit demo")
            self.win.Bind(_wx.EVT_MENU, self._OnExitApp, item)
            self.menubar.Append(menu, "&File")
            self.win.SetMenuBar(self.menubar)

        if not self._visible:
            self.visible = False
        if self._fullscreen:
            self.fullscreen = True

        rate.initialize(interactFunc=_Interact) # reset the rate data
        self.win.Raise()
        self.win.Show(True)

    def _add_display(self, disp):
        self._canvas = disp # this marks that there is a display that doesn't fill this window

    def delete(self):
        dl = _displays.displays[:]
        for d in dl:
            if d.win is self.win:
                for obj in d.objects:
                    obj.visible = False
                    del obj
                d.lights = []
                d.visible = False
                d._destroy()
        _displays.window_num -= 1
        self.win.DestroyChildren()
        self.win.Destroy()
        
    def _OnExitApp(self, evt):
        if self._canvas is None: # canvas isn't the only element of the window
            if self._exit:
                exit()
            else: # just close this window; leave other windows open
                self.win.DestroyChildren()
                self.win.Destroy()
        elif self._canvas.exit: # canvas fills the window; default is display.exit is True
            exit()
        else: # display.exit is False; just close this window; leave other windows open
            self.win.DestroyChildren()
            self.win.Destroy()
        _displays.window_num -= 1
        if _displays.window_num == 0:
            exit()
        else:
            for d in _displays.displays:
                if d.win is self.win:
                    d._destroy()

    def _OnMove(self, evt):
        c = self._canvas
        if c is not None:
            c._x, c._y = evt.GetPosition()
            c._x -= _dwidth//2
            c._y -= _dheight - _dwidth//2
            if self.menus: c._y -= _menuheight
            c._width = c.width
            c._height = c.height
            c._report_resize()
        _Interact()
        evt.Skip()
        
    def _OnSize(self, evt):
        c = self._canvas
        if c is not None:
            c._width, c._height = evt.GetSize()
            c._x = c.x
            c._y = c.y
            c._report_resize()
        if self._fullscreen_count == 2:
            self._fullscreen = False
            self._fullscreen_count = 0
        else:
            self._fullscreen_count += 1
        _Interact()
        evt.Skip()

    def _OnIconize(self, evt): # either iconized or restored
        self._visible = not self._visible

    def _OnMaximize(self, evt): # maximized (full screen); restore just drives _OnSize
        self._fullscreen_count = 1 # _OnSize is driven just after _OnMaximize
        self._fullscreen = True
    
    def _get_visible(self):
        return self._visible
    
    def _set_visible(self, v):
        if v:
            self.win.Restore()
        else:
            self.win.Iconize()
    
    def _get_fullscreen(self):
        return self._fullscreen
    
    def _set_fullscreen(self, full):
        if full:
            self.win.Maximize()
        else:
            self.win.Restore()

    visible = property( _get_visible, _set_visible )
    fullscreen = property( _get_fullscreen, _set_fullscreen )

class _mouseTracker:
    """
    mouseTracker is a simple class that's whole purpose in life is to
    keep track of the physical and logical state of the mouse so the
    display handler can focus on display. It's also easily separately
    testable this way.
    """

    def __init__(self):
        """
        leftDown, rightDown, middleDown are *physical* states
        spinning and zooming are *logical* states that depend
        on history and optional key-states. lastSpinning and
        lastZooming are used to check for the lock state when
        zooming and spinning are initiated.
        """
        self.leftIsDown = self.rightIsDown = self.middleIsDown = 0
        self.lastSpinning = self.lastZooming = 0
        self.macCtrl = 0  # mac ctrl is tricky..
        
    def macCtrlDown(self):
        self.macCtrl = 1  # we don't always get these ;-(. Only if the window has focus... dang.
        
    def macCtrlUp(self):
        if self.rightIsDown:
            self.rightUp()

        self.macCtrl = 0
        
    def leftDown(self):
        self.leftIsDown = 1
        
    def leftUp(self):
        self.leftIsDown = 0
        
    def rightDown(self):
        self.rightIsDown = 1
        
    def rightUp(self):
        self.rightIsDown = 0

    def midDown(self):
        self.middleIsDown = 1
        
    def midUp(self):
        self.middleIsDown = 0
        
    def isZooming(self, evt, userzoom=True, userspin=True):
        """
        Check to see if we're zooming. On a three button mouse
        this is done with the middle button. On a two button mouse
        this is done with both left/right buttons pressed simultaneously
        On a one button mouse this is done with the 'alt' key pressed along
        with the left button.
        """
        zooming = 0

        if not userzoom:
            return zooming
            
        if self.middleIsDown:
            zooming = 1
            
        elif self.leftIsDown and self.rightIsDown:
            zooming = 1
            
        elif self.leftIsDown and evt.AltDown():
            zooming = 1
            
        return zooming
        
    def isSpinning(self, evt, userzoom=True, userspin=True, zooming=False):
        """
        Check to see if we're spinning. One a two button mouse
        this is done with the right button UNLESS the left
        button is also down. So.. check for zooming first and 
        return False if we are. Then check for spinning and
        return accordingly.
        """
        
        spinning = 0
        if not userspin or zooming or self.isZooming(evt, userzoom, userspin):
            return spinning
            
        elif self.rightIsDown:
            spinning = 1
            
        elif self.leftIsDown and evt.CmdDown():
            #
            # For the mac we'll also take "cmd-left-click" to enable spinning
            #
            spinning = 1
            
        return spinning
        
    def checkLock(self, spinning, zooming):
        """
        Check to see if spinning and/or zooming has just
        been initiated. If so, return True, otherwise return False.
        """
        result = 0
        if spinning:
            if not self.lastSpinning:
                result = 1
                self.lastSpinning = 1
        else:
            self.lastSpinning = 0
            
        if zooming:
            if not self.lastZooming:
                result = 1
                self.lastZooming = 1
        else:
            self.lastZooming = 0
            
        return result
        
    def __repr__(self):
##      return '<%s lastZoom=%s, lastSpin=%s>' % (
##        self.__class__.__name__, `self.lastZooming`, `self.lastSpinning`)
      return '<%s lastZoom=%s, lastSpin=%s>' % (
        self.__class__.__name__, self.lastZooming, self.lastSpinning)
        
class binding_enabler(object):
    """
    just here to enable or disable bindings
    """
    def __init__(self, display, namestring, fn=None, enabled=True):
        self.display = display
        self.name = namestring
        self.fn = fn
        self.enabled = enabled
        
    def start(self):
        """
        try to support multiple events
        """
        if not self.enabled:
            self.enabled = True
            for name in self.name.split():
                self.display._enable_binding(name, self.fn, self.enabled)
        
    def stop(self):
        """
        try to support multiple events
        """
        if self.enabled:
            self.enabled = False
            for name in self.name.split():
                self.display._enable_binding(name, self.fn, self.enabled)

class eventInfo(object):
    """
    Generic object used to supply event information to event handlers
    in the users code.
    
    These can hold just about anything:
    
        obj = eventInfo(x=3, y=4, time=34, .....)
        
    produces and object with x, y and time attributes set accordingly.
    
    """
    def __init__(self, **kw):
        self.__dict__.update(kw)

class kb(object):
    def __init__(self):
        self.queue = []

    def pushkey(self, k):
        self.queue.append(k)

    def getkey(self):
        while len(self.queue) == 0:
            rate(30)
        ret = self.queue[0]
        self.queue = self.queue[1:]
        return ret

    def keys_in_queue(self):
        return len(self.queue)

    keys = property(keys_in_queue, None, None)

_legal_event_types = ['mousedown', 'mousemove', 'mouseup', 'click',
                      'keydown', 'keyup', 'char',
                      'redraw', 'draw_complete']

def set_cursor(win, visible):
        if visible:
            if 'phoenix' in _wx.PlatformInfo:
                win.SetCursor(_wx.Cursor(_wx.CURSOR_ARROW)) # Phoenix
            else:
                win.SetCursor(_wx.StockCursor(_wx.CURSOR_ARROW)) # Classic
        else:
            if 'phoenix' in _wx.PlatformInfo:
                win.SetCursor(_wx.Cursor(_wx.CURSOR_BLANK)) # Phoenix
            else:
                win.SetCursor(_wx.StockCursor(_wx.CURSOR_BLANK)) # Classic
    
class cursor(object): # used for both display.cursor.visible and window.cursor.visible
    def __init__(self, win=None, visible=True):
        self.win = win
        self._visible = visible

    def _get_visible(self):
        return self._visible

    def _set_visible(self, visible):
        self._visible = visible
        set_cursor(self.win, visible)

    visible = property(_get_visible, _set_visible)
                        
class display(cvisual.display_kernel):
    # wrap_display_kernel.cpp makes available to Python, in addition to
    # render_scene, report_mouse_state, and report_mouse_state,
    # the methods report_window_resize, report_view_resize, and pick.
            
    def __init__(self, **keywords):
        cvisual.display_kernel.__init__(self)
        self._window_initialized = False
        self.window = None
        self.menus = False
        self.N = -1
        self._lastx = self._lasty = None # previous mouse position
        self.fillswindow = True # assume canvas fills the window
        self._visible = True
        self._fullscreen = False
        
        self.material = materials.diffuse
        # If visible is set before width (say), can get error "can't change window".
        # So deal with visible attribute separately.
        v = None
        if 'visible' in keywords:
            self._visible = keywords['visible']
            del keywords['visible']
        if 'fullscreen' in keywords:
            self._fullscreen = keywords['fullscreen']
            del keywords['fullscreen']
        keys = list(keywords.keys())
        keys.sort()
        for kw in keys:
            setattr(self, kw, keywords[kw])
        if v is not None: setattr(self, 'visible', v)
        if 'ambient' not in keywords:
            self.ambient = (0.2,0.2,0.2)
        if self.window is not None:
            self.fillswindow = False
            self.win = self.window.win
            self.panel = self.window.panel

        if self.window is None:
            w = _screenwidth
            h = _screenheight
        else:
            w = self.window.width - self.window.dwidth
            h = self.window.height - self.window.dheight
            
        if self.x > w - 100:
            raise ValueError('display x too near the right edge of the screen')
        if self.x + self.width < 100:
            raise ValueError('display x too far to the left edge of the screen')
        if self.y > h - 100:
            raise ValueError('display y too close to the bottom of the screen')
        if self.y + self.height < 100:
            raise ValueError('display y too far to the top of the screen')

        if self.x + self.width > w:
            self.width = w - self.x
        if self.x < 0:
            self.width = self.x + self.width
            self.x = 0
        if self.y + self.height > h:
            self.height = h - self.y
        if self.y < 0:
            self.height = self.y + self.height
            self.y = 0
            
        self._mt = _mouseTracker()
        self._captured = 0
        self._cursorx = self._cursory = 0

        self.select()
        
        if 'lights' not in keywords:
            distant_light( direction=(0.22, 0.44, 0.88), color=(0.8,0.8,0.8) )
            distant_light( direction=(-0.88, -0.22, -.44), color=(0.3,0.3,0.3) )
            
        self._bindings = {}
        self._waiting = []
        self.keyboard = eventInfo()
        self.kb = kb()

    def select(self):
        cvisual.display_kernel.set_selected(self)

    def waitfor_event(self, evt):
        """
        driven by waitfor
        """
        self.waitfor_event_evt = evt
        self.waiting = False

    def waitfor(self, namestring):
        """
        wait for events listed in namestring, then proceed
        """
        names = namestring.split()
        s = ''
        for name in names:
            if name == 'redraw' or name == 'draw_complete' or name not in _legal_event_types:
                s += name+' or '
        if s != '':
            s = s[:-4]
            raise ValueError('waitfor event type cannot be '+s)
        self.waiting = True
        self.bind(namestring, self.waitfor_event)
        while self.waiting:
            rate(30)
        self.unbind(namestring, self.waitfor_event)
        return self.waitfor_event_evt
        
    def bind(self, namestring, fn, *args, **kw):
        """
        attach events listed in namestring to function 'fn'
        """
        names = namestring.split()
        enabled = True # start out enabled
        for name in names:
            elist = self._bindings.get(name,[])
            elist.append((enabled, fn, args, kw))  # each entry is a list [enabled, fn]
            self._bindings[name] = elist
                
        return binding_enabler(self, namestring, fn, enabled)
        
    def unbind(self, namestring, in_fn=None):
        """
        take a particular function out of binding, or all bindings
        from events in namestring if fn is None.
        """
        for name in namestring.split():
            new_list = []
            if in_fn is not None:
                for enabled, fn, args, kw in self._bindings.get(name,[]):
                    if in_fn != fn:
                        new_list.append((enabled, fn, args, kw))
                        
            if new_list:
                self._bindings[name] = new_list
            else:
                del self._bindings[name]
                
    def _enable_binding(self, name, in_fn, enable):
        new_list = []
        for enabled, fn, args, kw in self._bindings.get(name,[]):
            if in_fn is fn:
                new_list.append((enable, fn, args, kw))
            else:
                new_list.append((enabled, fn, args, kw))
                
        self._bindings[name] = new_list
                
    def _dispatch_event(self, name, evt=None, dargs=None, **kwArgs):
        if evt is not None:
            try:
                evt.event = name  # attempt to set the 'event' property to the name of the event
            except:
                pass
            
        for enabled, fn, args, kw in self._bindings.get(name,[]):
            if enabled:
                
                newKW = {}              # potential keyword arguments
                if evt is not None:
                    newArgs = (evt,) + args  # potential positional arguments
                else:
                    newArgs = args
                
                if dargs is not None:
                    newArgs += dargs   # args comes first... then dargs if not None
                                    
                argspec = getargspec(fn) # inspect the bound function and see what it's expecting
                
                newKW.update(kwArgs)
                if kw is not None:
                    newKW.update(kw)

                applyArgs = []
                if argspec.varargs is not None:
                    applyArgs = newArgs
                else:
                    if argspec.args is not None:
                        applyArgs = newArgs[:len(argspec.args)]
                
                if argspec.keywords is not None:
                    applyKW = newKW
                else:
                    applyKW = {}  # the keywords to actually apply to the bound function
                    [applyKW.update({name:newKW.get(name)}) for name in argspec.args if name is not 'self' and (name in newKW)] # populate the available keywords

                try:
                    # Not sure how to deal with the complexity of the applyKW dictionary, so ignore for now
                    fn(*applyArgs)
##                    apply(fn, applyArgs, applyKW)  # apply doesn't exist in Python 3
                except:
                    import traceback
                    traceback.print_exc()

    def trigger(self, namestring, *args, **kw):
        """
        trigger arbitrary events with keywords...
        """
        for name in namestring.split():
##            if args:
##                kw.update({'dargs':args})
##            apply(self._dispatch_event, (name,), kw)  # apply doesn't exist in Python 3
            self._dispatch_event(name, args[0])

    def _make_canvas(self, parent):
        # http://xoomer.virgilio.it/infinity77/wxPython/glcanvas/wx.glcanvas.GLCanvas.html
        # http://xoomer.virgilio.it/infinity77/wxPython/glcanvas/wx.glcanvas.GLContext.html

        x = y = 0
        w = self._width-_dwidth
        h = self._height-_dheight
        if _plat == 'Unix':
            y = 0
            h = h
        if self.fillswindow: # the canvas fills a window created by us
            if self.window._fullscreen:
                w, h = _wx.GetDisplaySize()
                self._width = w
                self._height = h
                self._x = self._y = 0
                self._report_resize()
        else: # place canvas in window already created by user
            x = self._x
            y = self._y
            w = self._width
            h = self._height
        # For backward compatibility, maintain self._canvas as well as self.canvas

        attribList = [_wx.glcanvas.WX_GL_DEPTH_SIZE, 24,
                      _wx.glcanvas.WX_GL_DOUBLEBUFFER, 1,
                      0]
        if _wx.glcanvas.GLCanvas_IsDisplaySupported(attribList):
            c = self.canvas = self._canvas = _wx.glcanvas.GLCanvas(parent, -1, pos=(x, y), size=(w, h), attribList = attribList)
        else:
            attribList = [_wx.glcanvas.WX_GL_DEPTH_SIZE, 16,
                          _wx.glcanvas.WX_GL_DOUBLEBUFFER, 1,
                          0]
            if _wx.glcanvas.GLCanvas_IsDisplaySupported(attribList):
                c = self.canvas = self._canvas = _wx.glcanvas.GLCanvas(parent, -1, pos=(x, y), size=(w, h), attribList = attribList)
            else:
                c = self.canvas = self._canvas = _wx.glcanvas.GLCanvas(parent, -1, pos=(x, y), size=(w, h))

        self._context = _wx.glcanvas.GLContext(c)

        if self.fillswindow: c.SetFocus()
        
        c.Bind(_wx.EVT_LEFT_DOWN, self._OnLeftMouseDown)
        c.Bind(_wx.EVT_LEFT_UP, self._OnLeftMouseUp)
        c.Bind(_wx.EVT_MIDDLE_DOWN, self._OnMiddleMouseDown)
        c.Bind(_wx.EVT_MIDDLE_UP, self._OnMiddleMouseUp)
        c.Bind(_wx.EVT_RIGHT_DOWN, self._OnRightMouseDown)
        c.Bind(_wx.EVT_RIGHT_UP, self._OnRightMouseUp)
        c.Bind(_wx.EVT_MOTION, self._OnMouseMotion)
        c.Bind(_wx.EVT_LEFT_DCLICK, self._OnLeftDClick)
        c.Bind(_wx.EVT_RIGHT_DCLICK, self._OnRightDClick)
        c.Bind(_wx.EVT_MIDDLE_DCLICK, self._OnMiddleDClick)
        c.Bind(_wx.EVT_MOUSE_CAPTURE_LOST, self._OnCaptureLost)
        
        #c.Bind(_wx.EVT_MOUSEWHEEL, self.OnMouseWheel)

        #c.Bind(_wx.EVT_CHAR, self._OnCharEvent)
        c.Bind(_wx.EVT_KEY_DOWN, self._OnKeyDown)
        c.Bind(_wx.EVT_KEY_UP, self._OnKeyUp)
        
    def _report_resize(self):
        self.report_window_resize(int(self._x), int(self._y), int(self._width), int(self._height))
        w = self._width-_dwidth
        if _plat == 'Unix':
            h = self._height-_menuheight
        else:
            h = self._height-_dheight
        if self.menus: h -= _menuheight
        if not self.fillswindow: # canvas was placed in user-created window
            w = self._width
            h = self._height
        self.report_view_resize(int(w), int(h))
        
    def _activate( self, a ): # this is called from C++ code in display_kernel.cpp
        self._activated = a
        if a:
            global _do_loop
            _do_loop = True # if a display has been activated, execute wait loop at exit
            _displays.add(self)       
            if not self._window_initialized:
                self._create()
                self._window_initialized = True         

    def _create(self):
        _displays.display_num += 1
        self.N = _displays.display_num
        self._x = self.x
        self._y = self.y
        self._width = self.width
        self._height = self.height
        if self.fillswindow: # the display fills the window; no panel created
            self.window = window(menus=self.menus, _make_panel=False, x=self._x, y=self._y,
                                 width=self._width, height=self._height, title=self.title,
                                 visible=self._visible, fullscreen=self._fullscreen)
            self.window._add_display(self)
            self._make_canvas(self.window.win)
        else:
            self._make_canvas(self.window.panel)
        self.cursor = cursor(win=self.canvas)
        self.win = self.window.win
           
        self.panel = self.window.panel
        self.menubar = self.window.menubar
        self._report_resize()
        _Interact()

    def delete(self):
        if self.fillswindow:
            self.window.delete()
        else:
            for obj in self.objects:
                obj.visible = False
                del obj
            self.lights = []
            _displays.remove(self)
            self.report_closed()
            self.canvas.Destroy()

    def _destroy(self):
        _displays.remove(self)
        self.win.Destroy()
        self.win = None
        self.report_closed()
        
    def _paint(self):
        if not self._window_initialized: return
        self._canvas.SetCurrent(self._context)
        self.render_scene()
        self._canvas.SwapBuffers()

## mouse event codes: http://docs.wxwidgets.org/2.8.4/wx_wxmouseevent.html#wxmouseevent

    def _OnLeftDClick(self, evt):
        self._OnLeftMouseDown(evt)
        self._OnLeftMouseUp(evt)
        self._OnLeftMouseDown(evt)
        
    def _OnRightDClick(self, evt):
        self._OnRightMouseDown(evt)
        self._OnRightMouseUp(evt)
        self._OnRightMouseDown(evt)

    def _OnMiddleDClick(self, evt):
        self._OnMiddleMouseDown(evt)
        self._OnMiddleMouseUp(evt)
        self._OnMiddleMouseDown(evt)
        
    def _OnLeftMouseDown(self, evt):
        self._mt.leftDown()
        self._report_mouse_state(evt)
        evt.Skip() # to permit setting focus

    def _OnLeftMouseUp(self, evt):
        self._mt.leftUp()
        self._report_mouse_state(evt)

    def _OnRightMouseDown(self, evt):
        self._mt.rightDown()
        self._report_mouse_state(evt)
        evt.Skip() # to permit setting focus

    def _OnRightMouseUp(self, evt):
        self._mt.rightUp()
        self._report_mouse_state(evt)

    def _OnMiddleMouseDown(self, evt):
        self._mt.midDown()
        self._report_mouse_state(evt)
        evt.Skip() # to permit setting focus

    def _OnMiddleMouseUp(self, evt):
        self._mt.midUp()
        self._report_mouse_state(evt)

##    def OnMouseWheel(self, evt): # not supported by VPython 5.x
##        print(evt.GetWheelRotation(), evt.GetWheelDelta())

    def _OnMouseMotion(self, evt):
        x, y = evt.GetPosition()
        if x != self._lastx or y != self._lasty:
            self._report_mouse_state(evt)
            self._dispatch_event('mousemove', self.mouse)
        evt.Skip() # to permit setting focus

    def _OnCaptureLost(self, evt):
        pass
            
# Mouse state, including ctrl/alt etc. http://wxpython.org/docs/api/wx.MouseState-class.html
# On Mac:
# CTRL + 1-button mouse = no CTRL, right button, presumably because CTRL-mouse == "right button"
# ALT  + 1-button mouse = ALT, left button
# CMD  + 1-button mouse = CTRL and CMD, left button

    def _report_mouse_state(self, evt): # wx gives x,y relative to upper left corner
        x, y = evt.GetPosition()
        if self._lastx is None:
            self._lastx = x
            self._lasty = y

        zooming = self._mt.isZooming(evt, self.userzoom, self.userspin)
        spinning = self._mt.isSpinning(evt, self.userzoom, self.userspin, zooming)
        lock = self._mt.checkLock(spinning, zooming)
        
        if lock and not self._captured:
            self.cursor_state = self.cursor.visible
            set_cursor(self.canvas, False)
            if self.fillswindow:
                self._cursorx, self._cursory = (x, y)
            else:
                # cursor is based on (0,0) of the window; our (x,y) is based on (0,0) of the 3D display
                self._cursorx, self._cursory = (int(self._x)+x, int(self._y)+y)
            self._canvas.CaptureMouse()
            self._captured = True
        elif self._captured and not (spinning or zooming):
            self.win.WarpPointer(self._cursorx, self._cursory)
            self._lastx = x = self._cursorx
            self._lasty = y = self._cursory
            set_cursor(self.canvas, self.cursor_state)
            self._canvas.ReleaseMouse()          
            self._captured = False
        
        
        #
        # So... we're going to report left/right/middle
        #

        left = self._mt.leftIsDown and not spinning and not zooming
        right = spinning or self._mt.rightIsDown
        middle = zooming or self._mt.middleIsDown
        shift = evt.ShiftDown()
        ctrl = evt.ControlDown()
        alt = evt.AltDown()
        cmd = evt.CmdDown()
                
        if _plat == 'Macintosh' and ctrl and cmd:
            #
            # Weird... if the user holds the cmd key, evt.ControlDown() returns True even if it's a lie.
            # So... we don't know if it's *really* down or not. ;-(
            #
            ctrl = False
        
#         labels = [s.strip() for s in "x, y, left, middle, right, shift, ctrl, alt, cmd, spin, zoom, lock, cap".split(',')]
#         vals = (x, y, left, middle, right, shift, ctrl, alt, cmd, spinning, zooming, lock, self._captured)
#         fmts = ["%9s"]*len(vals)
#         for l,f in zip(labels,fmts):
#             print(f % l, end='')
#         print()
#         for v,f in zip(vals,fmts):
#             print(f % `v`, end='')
#         print()
##        if trigger == 'leftdown' and not self._rightdown:
##            if ctrl:
##                right = 1
##                left = 0
##            elif alt:
##                middle = 1
##                left = 0

#        if (spinning or zooming) and (x == self._lastx) and (y == self._lasty): return
        
        self.report_mouse_state([left, right, middle],
                self._lastx, self._lasty, x, y,
                [shift, ctrl, alt, cmd])
        # For some reason, handling spin/zoom in terms of movements away
        # from a fixed cursor position fails on the Mac. As you drag the
        # mouse, repeated move mouse events mostly give the fixed cursor position.
        # Hence, for now, dragging off-screen stops spin/zoom on the Mac.
        # Similar problems on Ubuntu 12.04, plus wx.CURSOR_BLANK not available on Linux.
        if (spinning or zooming) and (_plat != 'Macintosh'): # reset mouse to original location
            self.win.WarpPointer(self._cursorx, self._cursory)
            if self.fillswindow:
                self._lastx = self._cursorx
                self._lasty = self._cursory
            else:
                # cursor is based on (0,0) of the window; our (x,y) is based on (0,0) of the 3D display
                self._lastx = self._cursorx - int(self._x)
                self._lasty = self._cursory - int(self._y)
        else: 
            self._lastx = x
            self._lasty = y

    def _ProcessChar(self, evt):
        #
        # Process event to keystroke for Char events.
        #
        key = evt.GetKeyCode()
        shift = evt.ShiftDown()
        if key > 127:
            if   key == 310: k = 'break'
            elif key == 312: k = 'end'
            elif key == 313: k = 'home'
            elif key == 314: k = 'left'
            elif key == 315: k = 'up'
            elif key == 316: k = 'right'
            elif key == 317: k = 'down'
            elif key == 322: k = 'insert'
            elif key == 340: k = 'f1'
            elif key == 341: k = 'f2'
            elif key == 342: k = 'f3'
            elif key == 343: k = 'f4'
            elif key == 344: k = 'f5'
            elif key == 345: k = 'f6'
            elif key == 346: k = 'f7'
            elif key == 347: k = 'f8'
            elif key == 348: k = 'f9'
            elif key == 349: k = 'f10'
            elif key == 366: k = 'page up'
            elif key == 367: k = 'page down'
            else: k = 'invalid key'
        else:
            # key code 311 is the caps lock key
            if shift or _wx.GetKeyState(311): k = _shifted[key]
            else: k = _unshifted[key]
        if k == 'invalid key':
            return k
        self.keyboard.key = k
        self.keyboard.ctrl = evt.ControlDown()
        self.keyboard.alt = evt.AltDown()
        self.keyboard.shift = evt.ShiftDown()
        self.keyboard.cmd = evt.CmdDown()
        prefix = ''
        if self.keyboard.ctrl:
            prefix += 'ctrl+'
        if self.keyboard.alt:
            prefix += 'alt+'
        if self.keyboard.shift and len(k) != 1:
            prefix += 'shift+'
        if _plat == 'Macintosh' and self.keyboard.cmd:
            prefix += 'cmd+'
        return prefix+k
    
    def _OnKeyDown(self, evt):
        k = self._ProcessChar(evt)
        if k != 'invalid key':
            self.kb.pushkey(k)
            self._dispatch_event('keydown', self.keyboard)
        #print ("got keydown:", evt.GetKeyCode(), codeLookup.get(evt.GetKeyCode(),k))
        if evt.GetKeyCode() == VPY_MAC_CTRL and _plat=='Macintosh':
            self._mt.macCtrlDown()
        #evt.Skip()
        
    def _OnKeyUp(self, evt):
        k = self._ProcessChar(evt)
        if k != 'invalid key':
            # old key up didn't count as an event
            self._dispatch_event('keyup', self.keyboard)
        #print ("got keyup:",evt.GetKeyCode(), codeLookup.get(evt.GetKeyCode(),k))
        if evt.GetKeyCode() == VPY_MAC_CTRL and _plat=='Macintosh':
            self._mt.macCtrlUp()
        evt.Skip()

    def _return_objects(self):
        return tuple([ o for o in self._get_objects() if not isinstance(o, cvisual.light) ])
    objects = property( _return_objects, None, None)

    def _get_lights(self):
        # TODO: List comprehension used for Python 2.3 compatibility; replace with
        #   generator comprehension
        return tuple([ o for o in self._get_objects() if isinstance(o, cvisual.light) ])
    
    def _set_lights(self, n_lights):
        old_lights = self._get_lights()
        for lt in old_lights:
            lt.visible = False

        if (type(n_lights) is not list) and (type(n_lights) is not tuple):
            n_lights = [n_lights] # handles case of scene.lights = single light
        for lt in n_lights:
            if isinstance( lt, cvisual.light ):  #< TODO: should this be allowed?
                lt.display = self
                lt.visible = True
            else:
                lum = cvisual.vector(lt).mag
                distant_light( direction=cvisual.vector(lt).norm(),
                               color=(lum,lum,lum),
                               display=self )
    
    def _get_visible(self):
        if self.fillswindow:
            return self.window.visible
        else:
            raise AttributeError('When a graphics -display- is only part of a window, get -visible- of the window.')
    
    def _set_visible(self, v):
        if self.fillswindow:
            if self.window: # means that a window has been created for the display
                self.window.visible = v
            else:
                self._visible = v
        else:
            raise AttributeError('When a graphics -display- is only part of a window, set -visible- of the window.')
    
    def _get_fullscreen(self):
        if self.fillswindow:
            return self.window.fullscreen
        else:
            raise AttributeError('When a graphics -display- is only part of a window, get -fullscreen- of the window.')
    
    def _set_fullscreen(self, full):
        if self.fillswindow:
            if self.window: # means that a window has been created for the display
                self.window.fullscreen = full
            else:
                self._fullscreen = full
        else:
            raise AttributeError('When a graphics -display- is only part of a window, set -fullscreen- of the window.')

    visible = property( _get_visible, _set_visible )
    fullscreen = property( _get_fullscreen, _set_fullscreen )
        
    ambient = property( cvisual.display_kernel._get_ambient, cvisual.display_kernel._set_ambient)
    range = property( cvisual.display_kernel._get_range, cvisual.display_kernel._set_range)
    lights = property( _get_lights, _set_lights, None)

from vis.site_settings import enable_shaders
display.enable_shaders = enable_shaders

# This is an atexit handler so that programs remain 'running' after they've finished
# executing the code in the program body. For visual this is important so that the
# windows don't close before the user can interact with them.
# _do_loop is set to True if a window is created or a display is activated.
# Otherwise we exit immediately (pure calculations with no displays).

_do_loop = False

def _close_final(): # There is a window, or an activated display
    global _do_loop
    if _do_loop:
        _do_loop = False # make sure we don't trigger this twice
        while True: # at end of user program, wait for user to close the program
            rate(1)
    _wx.Exit()

_atexit.register(_close_final)
# The following is needed by Python 3, else running from vidle3/run.py
# doesn't drive _close_final:
_sys.exitfunc = _close_final 
    
class _ManageDisplays(): # a singleton
    def __init__(self):
        set_wait(wait)
        self.displays = []
        self.window_num = 0     
        self.display_num = 0

    def add(self, d):
        self.displays.append(d)

    def remove(self, d):
        dl = self.displays[:]
        self.displays = []
        for disp in dl:
            if disp is d: continue
            self.displays.append(disp)

    def paint_displays(self):
        for d in self.displays:
            d._paint()

_displays = _ManageDisplays()
_evtloop = _wx.GUIEventLoop()
_wx.EventLoop.SetActive(_evtloop)
_isMac = ('wxOSX' in _wx.PlatformInfo)

if _plat == 'Windows':
    # On Windows, the best timer is supposedly time.clock()
    _clock = _time.clock
else:
    # On most other platforms, the best timer is supposedly time.time()
    _clock = _time.time

##_lastInteract = None
##
### This thread watches to see whether the window has become inactive,
### typically due to a loop not containing a call to rate or sleep.
### The check for "_lastInteract is not None" means that a program
### that imports visual but doesn't actually create any 3D objects
### will not be killed.
### This scheme doesn't work without making other changes. For example,
### the example program stonehenge.py gets terminated because there is
### lengthy setup of the scene. _Interact() gets called by the creation
### of the canvas, which is triggered by the creation of a 3D object.
### Also, during a long window move or resize, this times out.
### Maybe there's no way to detect a loop without a rate call?
##from threading import Thread
##class _timer(Thread):
##    def run(self):
##        while True:
##            _time.sleep(0.5)
##            if _lastInteract is not None and _clock() >= _lastInteract+0.5:
##                print("""
##Program terminated because window has become inactive.
##Probably there is a loop without a rate or sleep function.""")
##                _wx.Exit()
##                
##_thr = _timer()
##_thr.start()

_mouse_binding_names = set(['mousedown','mouseup','click'])   # these are the events that we need to grab using display.mouse.getevents()

def _Interact():
# The essence of _Interact was provided by Robin Dunn, developer of wxPython

##    global _lastInteract
##    _lastInteract = _clock()

    # Detect changes to ball.pos.x, which are not caught by the C++ code that
    # drives primitives/trail_update (changes to ball.x are caught)
    for obj in trail_list:
        try:
            if obj.make_trail and obj.interval > 0 and not obj.updated:
                if len(obj.trail_object.pos) == 0:
                    obj.trail_object.append(pos=obj.pos)
                elif obj.pos != obj.trail_object.pos[-1]:
                    obj.trail_object.append(pos=obj.pos, retain=obj.retain)
        except:
            trail_list.remove(obj)

    for d in _displays.displays:
        d._dispatch_event("redraw")

    _displays.paint_displays()
    
    for d in _displays.displays:
        d._dispatch_event("draw_complete")
    
    while not _evtloop.Pending() and _evtloop.ProcessIdle(): pass
    if _wx.GetApp(): _wx.GetApp().ProcessPendingEvents()
    if _isMac and not _evtloop.Dispatch(): return
    # Currently on wxOSX Pending always returns true, so the
    # ProcessIdle above is not ever called. Call it here instead.
    if _isMac: _evtloop.ProcessIdle()    
    
    while True:
        checkAgain = False
        if _wx.GetApp() and _wx.GetApp().HasPendingEvents():
            _wx.GetApp().ProcessPendingEvents()
            checkAgain = True
        if not _isMac and _evtloop.Pending():
            _evtloop.Dispatch()
            checkAgain = True
        if not checkAgain:
            break

    for d in _displays.displays:
        events = set(d._bindings.keys())
        if _mouse_binding_names.intersection(events):
            #
            # only dispatch these if bindings are registered
            #
            while d.mouse.events:
                event = d.mouse.getevent()
                #
                # Depending on event type.. forward to event handlers
                #
                if event.press:
                    d._dispatch_event("mousedown", event)
                elif event.release:
                    d._dispatch_event("mouseup", event)
                if event.click:
                    d._dispatch_event("click", event)

from .rate_function import RateKeeper as _rk
rate = _rk(interactFunc=_Interact)

def sleep(dt):
    tend = _clock() + dt
    _Interact() # make sure we call _Interact at least once
    while _clock() < tend:
        rate(30)
        
    
##print('--------------')
### Note that Mac info is incorrect (Linux is same as Mac)
##print(_wx.SystemSettings.GetMetric(_wx.SYS_SCREEN_X))    # 1920, 1440
##print(_wx.SystemSettings.GetMetric(_wx.SYS_SCREEN_Y))    # 1200, 900
##print(_wx.SystemSettings.GetMetric(_wx.SYS_BORDER_X))    # 1, -1 (Windows, Mac)
##print(_wx.SystemSettings.GetMetric(_wx.SYS_BORDER_Y))    # 1, -1
##print(_wx.SystemSettings.GetMetric(_wx.SYS_FRAMESIZE_X)) # 8, -1
##print(_wx.SystemSettings.GetMetric(_wx.SYS_FRAMESIZE_Y)) # 8, -1
##print(_wx.SystemSettings.GetMetric(_wx.SYS_MENU_Y))      #  20, -1
##print(_wx.SystemSettings.GetMetric(_wx.SYS_CAPTION_Y))   #  22, -1
##print(_wx.SystemSettings.GetMetric(_wx.SYS_EDGE_X))      #   2, -1
##print(_wx.SystemSettings.GetMetric(_wx.SYS_EDGE_Y))      #   2, -1
##print(_wx.SystemSettings.GetMetric(_wx.SYS_WINDOWMIN_X)) # 132, -1
##print(_wx.SystemSettings.GetMetric(_wx.SYS_WINDOWMIN_Y)) #  38, -1
##print('--------------')

# Title bar height is FRAMESIZE_Y + CAPTION_Y = 30 on Windows
# Menu bar height is MENU_Y = 20 on Windows
# Thickness of bottom is FRAMESIZE_Y = 8 on Windows
#   So height of canvas is scene.height - 58
#   and width of canvas is scene.width - 16

# The canvas size is (self._width-_dwidth, self._height-_dheight)
# except for Xubuntu, where some ad hoc corrections had to be made.
# _dy is the number of pixels from the top of the window to the top of the display area
if _plat == 'Macintosh':
    _dwidth = 0
    _dheight = 22
    _menuheight = 0
elif _plat == 'Unix': # e.g., Linux
    # Xubuntu: title bar 49 high, menubar 27 high, mouse offset 52
    # Ubuntu 12.04: title bar 27, menubar ?, mouse offset ?
    _dwidth = 0
    _dheight = 27
    _menuheight = 27
else: # Windows
    _dwidth = 2*(_wx.SystemSettings.GetMetric(_wx.SYS_FRAMESIZE_X))
    _dheight = 2*(_wx.SystemSettings.GetMetric(_wx.SYS_FRAMESIZE_Y)) + \
            _wx.SystemSettings.GetMetric(_wx.SYS_CAPTION_Y)
    _menuheight = _wx.SystemSettings.GetMetric(_wx.SYS_FRAMESIZE_Y) + \
            _wx.SystemSettings.GetMetric(_wx.SYS_CAPTION_Y)

window.dwidth = _dwidth
window.dheight = _dheight
window.menuheight = _menuheight
        
##        # The canvas size can also be determined by comparing GetSize() and GetClientSize(),
##        # after the window and canvas have been created:
##        size = self.win.GetSize()
##        canvas_size = self.win.GetClientSize()
##        self._dwidth = size[0] - canvas_size[0] # canvas is dwidth thinner than scene.width
##        self._dheight = size[1] - canvas_size[1] # canvas is dheight shorter than scene.height

# Tables for converting US keyboard inputs to characters
_unshifted = ['', '', '', '', '', '', '', '', 'backspace', 'tab', # 0-9
   '', '', '', '\n', '', '', 'shift', 'ctrl', 'alt', '', # 10-19
   'caps lock', '', '', '', '', '', '', 'esc', '', '', #20-29
   '', '', ' ', '', '', '', '', '', '', "'", # 30-39
   '', '', '', '', ',', '-', '.', '/', '0', '1', # 40-49
   '2', '3', '4', '5', '6', '7', '8', '9', '', ';', # 50-59
   '', '=', '', '', '', 'a', 'b', 'c', 'd', 'e', # 60-69
   'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', # 70-79
   'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', # 80-89
   'z', '[', '\\', ']', '', '', '`', '', '', '', # 90-99
   '', '', '', '', '', '', '', '', '', '', # 100-109
   '', '', '', '', '', '', '', '', '', '', # 110-119
   '', '', '', '', '', '', '', 'delete'] #120-127
   
_shifted = ['', '', '', '', '', '', '', '', 'backspace', 'tab', # 0-9
   '', '', '', '\n', '', '', 'shift', 'ctrl', 'alt', 'break', # 10-19
   'caps lock', '', '', '', '', '', '', 'esc', '', '', #20-29
   '', '', '', '!', '"', '#', '$', '%', '&', '"', # 30-39
   '(', ')', '*', '+', '<', '_', '>', '?', ')', '!', # 40-49
   '@', '#', '$', '%', '^', '&', '*', '(', ':', ':', # 50-59
   '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', # 60-69
   'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', # 70-79
   'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', # 80-89
   'Z', '{', '|', '}', '^', '_', '~', '', '', '', # 90-99
   '', '', '', '', '', '', '*', '+', '', '', # 100-109
   '', '', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', # 110-119
   'f9', 'f10', '', '{', '|', '}', '~', 'delete'] #120-127

##    def _OnCharEvent(self, evt):
##        k = self._ProcessChar(evt)
##        self.kb.pushkey(k)
##        self.keyboard.key = k
##        self._dispatch_event('char', self.keyboard)
##        evt.Skip()
#
# The wx key codes:
#
# WXK_BACK     WXK_EXECUTE    WXK_F1   WXK_NUMPAD_SPACE      WXK_WINDOWS_LEFT
# WXK_TAB      WXK_SNAPSHOT   WXK_F2   WXK_NUMPAD_TAB        WXK_WINDOWS_RIGHT
# WXK_RETURN   WXK_INSERT     WXK_F3   WXK_NUMPAD_ENTER      WXK_WINDOWS_MENU
# WXK_ESCAPE   WXK_HELP       WXK_F4   WXK_NUMPAD_F1         WXK_SPECIAL1
# WXK_SPACE    WXK_NUMPAD0    WXK_F5   WXK_NUMPAD_F2         WXK_SPECIAL2
# WXK_DELETE   WXK_NUMPAD1    WXK_F6   WXK_NUMPAD_F3         WXK_SPECIAL3
# WXK_LBUTTON  WXK_NUMPAD2    WXK_F7   WXK_NUMPAD_F4         WXK_SPECIAL4
# WXK_RBUTTON  WXK_NUMPAD3    WXK_F8   WXK_NUMPAD_HOME       WXK_SPECIAL5
# WXK_CANCEL   WXK_NUMPAD4    WXK_F9   WXK_NUMPAD_LEFT       WXK_SPECIAL6
# WXK_MBUTTON  WXK_NUMPAD5    WXK_F10  WXK_NUMPAD_UP         WXK_SPECIAL7
# WXK_CLEAR    WXK_NUMPAD6    WXK_F11  WXK_NUMPAD_RIGHT      WXK_SPECIAL8
# WXK_SHIFT    WXK_NUMPAD7    WXK_F12  WXK_NUMPAD_DOWN       WXK_SPECIAL9
# WXK_ALT      WXK_NUMPAD8    WXK_F13  WXK_NUMPAD_PRIOR      WXK_SPECIAL10
# WXK_CONTROL  WXK_NUMPAD9    WXK_F14  WXK_NUMPAD_PAGEUP     WXK_SPECIAL11
# WXK_MENU     WXK_MULTIPLY   WXK_F15  WXK_NUMPAD_NEXT       WXK_SPECIAL12
# WXK_PAUSE    WXK_ADD        WXK_F16  WXK_NUMPAD_PAGEDOWN   WXK_SPECIAL13
# WXK_CAPITAL  WXK_SEPARATOR  WXK_F17  WXK_NUMPAD_END        WXK_SPECIAL14
# WXK_PRIOR    WXK_SUBTRACT   WXK_F18  WXK_NUMPAD_BEGIN      WXK_SPECIAL15
# WXK_NEXT     WXK_DECIMAL    WXK_F19  WXK_NUMPAD_INSERT     WXK_SPECIAL16
# WXK_END      WXK_DIVIDE     WXK_F20  WXK_NUMPAD_DELETE     WXK_SPECIAL17
# WXK_HOME     WXK_NUMLOCK    WXK_F21  WXK_NUMPAD_EQUAL      WXK_SPECIAL18
# WXK_LEFT     WXK_SCROLL     WXK_F22  WXK_NUMPAD_MULTIPLY   WXK_SPECIAL19
# WXK_UP       WXK_PAGEUP     WXK_F23  WXK_NUMPAD_ADD        WXK_SPECIAL20
# WXK_RIGHT    WXK_PAGEDOWN   WXK_F24  WXK_NUMPAD_SEPARATOR   
# WXK_DOWN                             WXK_NUMPAD_SUBTRACT    
# WXK_SELECT                           WXK_NUMPAD_DECIMAL     
# WXK_PRINT                            WXK_NUMPAD_DIVIDE
# 
# With time we can include all of these..
#

codeStrings = [x.strip() for x in """
WXK_BACK     WXK_EXECUTE    WXK_F1   WXK_NUMPAD_SPACE      WXK_WINDOWS_LEFT
WXK_TAB      WXK_SNAPSHOT   WXK_F2   WXK_NUMPAD_TAB        WXK_WINDOWS_RIGHT
WXK_RETURN   WXK_INSERT     WXK_F3   WXK_NUMPAD_ENTER      WXK_WINDOWS_MENU
WXK_ESCAPE   WXK_HELP       WXK_F4   WXK_NUMPAD_F1         WXK_SPECIAL1
WXK_SPACE    WXK_NUMPAD0    WXK_F5   WXK_NUMPAD_F2         WXK_SPECIAL2
WXK_DELETE   WXK_NUMPAD1    WXK_F6   WXK_NUMPAD_F3         WXK_SPECIAL3
WXK_LBUTTON  WXK_NUMPAD2    WXK_F7   WXK_NUMPAD_F4         WXK_SPECIAL4
WXK_RBUTTON  WXK_NUMPAD3    WXK_F8   WXK_NUMPAD_HOME       WXK_SPECIAL5
WXK_CANCEL   WXK_NUMPAD4    WXK_F9   WXK_NUMPAD_LEFT       WXK_SPECIAL6
WXK_MBUTTON  WXK_NUMPAD5    WXK_F10  WXK_NUMPAD_UP         WXK_SPECIAL7
WXK_CLEAR    WXK_NUMPAD6    WXK_F11  WXK_NUMPAD_RIGHT      WXK_SPECIAL8
WXK_SHIFT    WXK_NUMPAD7    WXK_F12  WXK_NUMPAD_DOWN       WXK_SPECIAL9
WXK_ALT      WXK_NUMPAD8    WXK_F13  WXK_NUMPAD_PRIOR      WXK_SPECIAL10
WXK_CONTROL  WXK_NUMPAD9    WXK_F14  WXK_NUMPAD_PAGEUP     WXK_SPECIAL11
WXK_MENU     WXK_MULTIPLY   WXK_F15  WXK_NUMPAD_NEXT       WXK_SPECIAL12
WXK_PAUSE    WXK_ADD        WXK_F16  WXK_NUMPAD_PAGEDOWN   WXK_SPECIAL13
WXK_CAPITAL  WXK_SEPARATOR  WXK_F17  WXK_NUMPAD_END        WXK_SPECIAL14
WXK_PRIOR    WXK_SUBTRACT   WXK_F18  WXK_NUMPAD_BEGIN      WXK_SPECIAL15
WXK_NEXT     WXK_DECIMAL    WXK_F19  WXK_NUMPAD_INSERT     WXK_SPECIAL16
WXK_END      WXK_DIVIDE     WXK_F20  WXK_NUMPAD_DELETE     WXK_SPECIAL17
WXK_HOME     WXK_NUMLOCK    WXK_F21  WXK_NUMPAD_EQUAL      WXK_SPECIAL18
WXK_LEFT     WXK_SCROLL     WXK_F22  WXK_NUMPAD_MULTIPLY   WXK_SPECIAL19
WXK_UP       WXK_PAGEUP     WXK_F23  WXK_NUMPAD_ADD        WXK_SPECIAL20
WXK_RIGHT    WXK_PAGEDOWN   WXK_F24  WXK_NUMPAD_SEPARATOR   
WXK_DOWN                             WXK_NUMPAD_SUBTRACT    
WXK_SELECT                           WXK_NUMPAD_DECIMAL     
WXK_PRINT                            WXK_NUMPAD_DIVIDE VPY_MAC_CTRL
""".split()]  

#
# the mac 'ctrl' key seems to come in as key code 396, not in WXK list.
# the mac 'cmd' key comes through as WXK_CTRL, oddly enough.
#

VPY_MAC_CTRL = 396

##codeVals = [
##_wx.WXK_BACK,_wx.WXK_EXECUTE,_wx.WXK_F1,_wx.WXK_NUMPAD_SPACE,_wx.WXK_WINDOWS_LEFT,
##_wx.WXK_TAB,_wx.WXK_SNAPSHOT,_wx.WXK_F2,_wx.WXK_NUMPAD_TAB,_wx.WXK_WINDOWS_RIGHT,
##_wx.WXK_RETURN,_wx.WXK_INSERT,_wx.WXK_F3,_wx.WXK_NUMPAD_ENTER,_wx.WXK_WINDOWS_MENU,
##_wx.WXK_ESCAPE,_wx.WXK_HELP,_wx.WXK_F4,_wx.WXK_NUMPAD_F1,_wx.WXK_SPECIAL1,
##_wx.WXK_SPACE,_wx.WXK_NUMPAD0,_wx.WXK_F5,_wx.WXK_NUMPAD_F2,_wx.WXK_SPECIAL2,
##_wx.WXK_DELETE,_wx.WXK_NUMPAD1,_wx.WXK_F6,_wx.WXK_NUMPAD_F3,_wx.WXK_SPECIAL3,
##_wx.WXK_LBUTTON,_wx.WXK_NUMPAD2,_wx.WXK_F7,_wx.WXK_NUMPAD_F4,_wx.WXK_SPECIAL4,
##_wx.WXK_RBUTTON,_wx.WXK_NUMPAD3,_wx.WXK_F8,_wx.WXK_NUMPAD_HOME,_wx.WXK_SPECIAL5,
##_wx.WXK_CANCEL,_wx.WXK_NUMPAD4,_wx.WXK_F9,_wx.WXK_NUMPAD_LEFT,_wx.WXK_SPECIAL6,
##_wx.WXK_MBUTTON,_wx.WXK_NUMPAD5,_wx.WXK_F10,_wx.WXK_NUMPAD_UP,_wx.WXK_SPECIAL7,
##_wx.WXK_CLEAR,_wx.WXK_NUMPAD6,_wx.WXK_F11,_wx.WXK_NUMPAD_RIGHT,_wx.WXK_SPECIAL8,
##_wx.WXK_SHIFT,_wx.WXK_NUMPAD7,_wx.WXK_F12,_wx.WXK_NUMPAD_DOWN,_wx.WXK_SPECIAL9,
##_wx.WXK_ALT,_wx.WXK_NUMPAD8,_wx.WXK_F13,_wx.WXK_NUMPAD_PRIOR,_wx.WXK_SPECIAL10,
##_wx.WXK_CONTROL,_wx.WXK_NUMPAD9,_wx.WXK_F14,_wx.WXK_NUMPAD_PAGEUP,_wx.WXK_SPECIAL11,
##_wx.WXK_MENU,_wx.WXK_MULTIPLY,_wx.WXK_F15,_wx.WXK_NUMPAD_NEXT,_wx.WXK_SPECIAL12,
##_wx.WXK_PAUSE,_wx.WXK_ADD,_wx.WXK_F16,_wx.WXK_NUMPAD_PAGEDOWN,_wx.WXK_SPECIAL13,
##_wx.WXK_CAPITAL,_wx.WXK_SEPARATOR,_wx.WXK_F17,_wx.WXK_NUMPAD_END,_wx.WXK_SPECIAL14,
##_wx.WXK_PRIOR,_wx.WXK_SUBTRACT,_wx.WXK_F18,_wx.WXK_NUMPAD_BEGIN,_wx.WXK_SPECIAL15,
##_wx.WXK_NEXT,_wx.WXK_DECIMAL,_wx.WXK_F19,_wx.WXK_NUMPAD_INSERT,_wx.WXK_SPECIAL16,
##_wx.WXK_END,_wx.WXK_DIVIDE,_wx.WXK_F20,_wx.WXK_NUMPAD_DELETE,_wx.WXK_SPECIAL17,
##_wx.WXK_HOME,_wx.WXK_NUMLOCK,_wx.WXK_F21,_wx.WXK_NUMPAD_EQUAL,_wx.WXK_SPECIAL18,
##_wx.WXK_LEFT,_wx.WXK_SCROLL,_wx.WXK_F22,_wx.WXK_NUMPAD_MULTIPLY,_wx.WXK_SPECIAL19,
##_wx.WXK_UP,_wx.WXK_PAGEUP,_wx.WXK_F23,_wx.WXK_NUMPAD_ADD,_wx.WXK_SPECIAL20,
##_wx.WXK_RIGHT,_wx.WXK_PAGEDOWN,_wx.WXK_F24,_wx.WXK_NUMPAD_SEPARATOR,
##_wx.WXK_DOWN,_wx.WXK_NUMPAD_SUBTRACT,
##_wx.WXK_SELECT,_wx.WXK_NUMPAD_DECIMAL,
##_wx.WXK_PRINT,_wx.WXK_NUMPAD_DIVIDE,VPY_MAC_CTRL
##]
##
##codeLookup = dict(zip(codeVals, codeStrings))
