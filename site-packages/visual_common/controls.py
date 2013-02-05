from .cvisual import vector, mag, norm, dot
from .primitives import (box, cylinder, distant_light, frame, label, sphere)
from .create_display import display
from . import crayola
color = crayola
from math import pi, sin, cos
import time

# Bruce Sherwood, begun March 2002
# Import this module to create buttons, toggle switches, sliders, and pull-down menus.
# See controlstest.py in the VPython example programs for an example of how to use controls.
        
lastcontrols = None # the most recently created controls window
gray = (0.7, 0.7, 0.7)
darkgray = (0.5, 0.5, 0.5)

cdisplays = []

def checkControlsMouse(evt, c):
        try:
            if c.display.visible: c.mouse(evt)
        except:
            pass
    
class controls: # make a special window for buttons, sliders, and pull-down menus
    def __init__(self, x=0, y=0, width=300, height=320, range=100,
                 title=None, foreground=None, background=None):
        global lastcontrols
        global cdisplays
        lastcontrols = self
        currentdisplay = display.get_selected()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.range = range
        self.title = title
        if title is None:
            title = 'Controls'
        if foreground is None:
            foreground = color.white
        if background is None:
            background = color.black
        self.foreground = foreground
        self.background = background
        self.display = display(title=title, x=x, y=y, range=range,
                width=width, height=height, fov=0.4,
                foreground=foreground, background=background,
                userzoom=0, userspin=0)
        self.display.lights=[distant_light(direction=(0,0,1),color=color.white)]
        self.focus = None
        self.lastpos = None
        self.controllist = []
        currentdisplay.select()
        cdisplays.append(self)
        self.display.bind("mousedown", checkControlsMouse, self)
        self.display.bind("mousemove", checkControlsMouse, self)
        self.display.bind("mouseup", checkControlsMouse, self)

    def __del__(self):
        self.visible = False
        try:
            cdisplays.remove(self)
        except:
            pass

    def addcontrol(self, control):
        self.controllist.append(control)

    def interact(self):
        # For backward compatibility; no longer necessary to call interact()
        pass
    
    def mouse(self, evt): 
        m = evt
        if m.press == 'left' and m.pick:
            picked = m.pick
            if self.focus: # have been moving over menu with mouse up
                picked = self.focus
            for control in self.controllist:
                if control.active is picked:
                    self.focus = control
                    control.highlight(m.pos)
        elif m.release == 'left':
            focus = self.focus
            self.focus = None # menu may reset self.focus for "sticky" menu
            if focus:
                focus.unhighlight(m.pos)
        if self.focus: # if dragging a control
            pos = self.display.mouse.pos
            if pos != self.lastpos:
                self.focus.update(pos)
                self.lastpos = pos

class ctrl(object): # common aspects of buttons, sliders, and menus
    # Note: ctrl is a subclass of "object" in order to be a new-type class which
    # permits use of the new "property" feature exploited by buttons and sliders.
    def __init__(self, args):
        if 'controls' in args:
            self.controls = args['controls']
        elif lastcontrols is None:
            self.controls = controls()
        else:
            self.controls = lastcontrols
        self.controls.addcontrol(self)
        self.pos = vector(0,0)
        self.action = None
        if 'pos' in args:
            self.pos = vector(args['pos'])
        if 'value' in args:
            self.value = args['value']
        if 'action' in args:
            self.action = args['action']
            
    def highlight(self, pos):
        pass
        
    def unhighlight(self, pos):
        pass

    def update(self, pos):
        pass
        
    def execute(self):
        if self.action:
            self.action()

class button(ctrl):
    def __init__(self, **args):
        self.type = 'button'
        self.value = 0
        ctrl.__init__(self, args)
        width = height = 40
        bcolor = gray
        edge = darkgray
        self.__text = ''
        if 'width' in args:
            width = args['width']
        if 'height' in args:
            height = args['height']
        if 'text' in args:
            self.__text = args['text']
        if 'color' in args:
            bcolor = args['color']
        disp = self.controls.display
        framewidth = width/10.
        self.thick = 2.*framewidth
        self.box1 = box(display=disp, pos=self.pos+vector(0,height/2.-framewidth/2.,0),
                       size=(width,framewidth,self.thick), color=edge)
        self.box2 = box(display=disp, pos=self.pos+vector(-width/2.+framewidth/2.,0,0),
                       size=(framewidth,height,self.thick), color=edge)
        self.box3 = box(display=disp, pos=self.pos+vector(width/2.-framewidth/2.,0,0),
                       size=(framewidth,height,self.thick), color=edge)
        self.box4 = box(display=disp, pos=self.pos+vector(0,-height/2.+framewidth/2.,0),
                       size=(width,framewidth,self.thick), color=edge)
        self.button = box(display=disp, pos=self.pos+vector(0,0,self.thick/2.+1.),
                       size=(width-2.*framewidth,height-2.*framewidth,self.thick), color=bcolor)
        self.label = label(display=disp, pos=self.button.pos, color=color.black,
                           text=self.__text, line=0, box=0, opacity=0)
        self.active = self.button

    def gettext(self):
        return self.label.text

    def settext(self, text):
        self.label.text = text

    text = property(gettext, settext) # establishes special getattr/setattr handling

    def highlight(self, pos):
        self.button.pos.z -= self.thick
        self.label.pos.z -= self.thick
        self.value = 1
        
    def unhighlight(self, pos):
        self.button.pos.z += self.thick
        self.label.pos.z += self.thick
        self.value = 0
        self.execute()

class toggle(ctrl):
    def __init__(self, **args):
        self.type = 'toggle'
        self.__value = 0
        ctrl.__init__(self, args)
        width = height = 20
        self.angle = pi/6. # max rotation of toggle
        bcolor = gray
        edge = darkgray
        self.__text0 = ''
        self.__text1 = ''
        if 'width' in args:
            width = args['width']
        if 'height' in args:
            height = args['height']
        if 'text0' in args:
            self.__text0 = args['text0']
        if 'text1' in args:
            self.__text1 = args['text1']
        if 'color' in args:
            bcolor = args['color']
        if 'value' in args:
            self.__value = args['value']
        diskthick = width/4.
        diskradius = height/2.
        ballradius = 0.6*diskradius
        self.rodlength = 1.2*diskradius+ballradius
        disp = self.controls.display
        self.frame = frame(display=disp, pos=self.pos, axis=(1,0,0))
        self.back = box(display=disp, frame=self.frame, pos=(0,0,0),
                        size=(width,height,0.3*diskradius), color=darkgray)
        self.disk1 = cylinder(display=disp, frame=self.frame, pos=(-diskthick,0,0),
                              axis=(-diskthick,0), radius=diskradius, color=gray)
        self.disk2 = cylinder(display=disp, frame=self.frame, pos=(diskthick,0,0),
                              axis=(diskthick,0), radius=diskradius, color=gray)
        self.rod = cylinder(display=disp, frame=self.frame, pos=(0,0,0),
                              axis=(0,0,self.rodlength), radius=width/8., color=gray)
        self.ball = sphere(display=disp, frame=self.frame, pos=(0,0,self.rodlength),
                              radius=ballradius, color=gray)
        self.label0 = label(display=disp, frame=self.frame, pos=(0,-1.0*height), text=self.__text0,
                           line=0, box=0, opacity=0)
        self.label1 = label(display=disp, frame=self.frame, pos=(0,1.0*height), text=self.__text1,
                           line=0, box=0, opacity=0)
        self.settoggle(self.__value)
        self.active = self.ball

    def settoggle(self, val):
        self.__value = val
        if val == 1:
            newpos = self.rodlength*vector(0,sin(self.angle), cos(self.angle))
        else:
            newpos = self.rodlength*vector(0,-sin(self.angle), cos(self.angle))
        self.rod.axis = newpos
        self.ball.pos = newpos
             
    def getvalue(self):
        return self.__value

    def setvalue(self, val):
        self.settoggle(val)
        self.__value = val

    value = property(getvalue, setvalue) # establishes special getattr/setattr handling

    def gettext0(self):
        return self.label0.text

    def settext0(self, text):
        self.label0.text = text

    text0 = property(gettext0, settext0) # establishes special getattr/setattr handling

    def gettext1(self):
        return self.label1.text

    def settext1(self, text):
        self.label1.text = text

    text1 = property(gettext1, settext1) # establishes special getattr/setattr handling
        
    def unhighlight(self, pos):
        if self.controls.display.mouse.pick is self.active:
            self.__value = not(self.__value)
            self.settoggle(self.__value)
            self.execute()

class slider(ctrl):
    def __init__(self, **args):
        self.type = 'slider'
        self.canupdate = False # cannot update slider position until all attributes initialized
        self.__value = 0
        self.length = 100.
        width = 10.
        shaftcolor = darkgray
        scolor = gray
        self.min = 0.
        self.max = 100.
        self.axis = vector(1,0,0)
        if 'axis' in args:
            self.axis = vector(args['axis'])
            self.length = mag(self.axis)
            self.axis = norm(self.axis)
        if 'length' in args:
            self.length = args['length']
        if 'width' in args:
            width = args['width']
        if 'min' in args:
            self.min = args['min']
            if self.__value < self.min:
                self.__value = self.min
        if 'max' in args:
            self.max = args['max']
            if self.__value > self.max:
                self.__value = self.max
        if 'color' in args:
            scolor = args['color']
        ctrl.__init__(self, args)
        disp = self.controls.display
        self.shaft = box(display=disp, 
                       pos=self.pos+self.axis*self.length/2., axis=self.axis,
                       size=(self.length,0.5*width,0.5*width), color=shaftcolor)
        self.indicator = box(display=disp,
                       pos=self.pos+self.axis*(self.__value-self.min)*self.length/(self.max-self.min),
                       axis=self.axis,
                       size=(width,width,width), color=scolor)
        self.active = self.indicator
        self.canupdate = True

    def getvalue(self):
        return self.__value

    def setvalue(self, val):
        if self.canupdate: self.update(self.pos+self.axis*(val-self.min)*self.length/(self.max-self.min))
        self.__value = val

    value = property(getvalue, setvalue) # establishes special getattr/setattr handling

    def update(self, pos):
        val = self.min+dot((pos-self.pos),self.axis)*(self.max-self.min)/self.length
        if val < self.min:
            val = self.min
        elif val > self.max:
            val = self.max
        if val != self.__value:
            self.indicator.pos = self.pos+self.axis*(val-self.min)*self.length/(self.max-self.min)
            self.__value = val
            self.execute()

class menu(ctrl):
    def __init__(self, **args):
        self.type = 'menu'
        ctrl.__init__(self, args)
        self.items = []
        self.width = self.height = 40
        self.text = 'Menu'
        self.__value = None
        self.color = gray
        self.nitem = 0
        self.open = 0 # true if menu display open in the window
        self.action = 1 # dummy placeholder; what is driven is menu.execute()
        if 'width' in args:
            self.width = args['width']
        if 'height' in args:
            self.height = args['height']
        if 'text' in args:
            self.text = args['text']
        if 'color' in args:
            self.color = args['color']
        self.thick = 0.2*self.width
        disp = self.controls.display
        self.active = box(display=disp, pos=self.pos+vector(0,0,self.thick),
                       size=(self.width,self.height,self.thick), color=self.color)
        self.label = label(display=disp, pos=self.active.pos, color=color.black,
                           text=self.text, line=0, box=0, opacity=0)

    def getvalue(self):
        return self.__value

    value = property(getvalue, None) # establishes special getattr/setattr handling

    def inmenu(self, pos): # return item number (0-N) where mouse is, or -1
        # note that item is 0 if mouse is in menu title
        if self.pos.x-self.width/2. < pos.x < self.pos.x+self.width/2.:
            nitem = int((self.pos.y+self.height/2.-pos.y)/self.height)
            if 0 <= nitem <= len(self.items):
                return(nitem)
            else:
                return(-1)
        return(-1)
        
    def highlight(self, pos): # mouse down: open the menu, displaying the menu items
        self.nitem = self.inmenu(pos)
        if self.open: # "sticky" menu already open
            if self.nitem > 0:
                self.update(pos)
            else:
                self.unhighlight(pos)
                self.open = 0
            return
        pos = self.pos-vector(0,self.height,0)
        self.boxes = []
        self.highlightedbox = None
        disp = self.controls.display
        for item in self.items:
            self.boxes.append( (box(display=disp, pos=pos+vector(0,0,self.thick),
                       size=(self.width,self.height,self.thick), color=self.color),
                       label(display=disp, pos=pos+vector(0,0,self.thick), color=color.black,
                       text=item[0], line=0, box=0, opacity=0)) )
            pos = pos-vector(0,self.height,0)

    def unhighlight(self, pos): # mouse up: close the menu; selected item will be executed
        self.nitem = self.inmenu(pos)
        if self.nitem == 0 and not self.open: # don't close if mouse up in menu title
            self.controls.focus = self # restore menu to be in focus
            self.open = 1
            return
        for box in self.boxes:
            box[0].visible = 0
            box[1].visible = 0
        self.boxes = []
        self.open = 0
        self.execute()
            
    def update(self, pos): # highlight an individual item during drag
        self.nitem = self.inmenu(pos)
        if self.nitem > 0:
            if self.highlightedbox is not None:
                self.highlightedbox.color = gray
            if self.items[self.nitem-1][1]: # if there is an associated action
                self.highlightedbox = self.boxes[self.nitem-1][0]
                self.highlightedbox.color = darkgray
        else:
            if self.highlightedbox is not None:
                self.highlightedbox.color = gray
            self.highlightedbox = None

    def execute(self):
        if self.nitem > 0:
            self.__value = self.items[self.nitem-1][0]
            action = self.items[self.nitem-1][1]
            if action:
                action()

       
