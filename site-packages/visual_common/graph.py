from __future__ import division
from .cvisual import vector
from .primitives import (label, curve, faces, points, distant_light)
from .create_display import display
from . import crayola

color = crayola
from numpy import (array, arange, ndarray, zeros, sort, searchsorted,
                   concatenate)
from math import modf, log10
import time

# A graph package for plotting a curve, with labeled axes and autoscaling
# Bruce Sherwood, begun April 2000
# Added crosshairs March 2011; added log-log and semilog plots April 2011

gdisplays = []

def checkGraphMouse(evt, gd):
    try:
        gd.mouse(evt)
    except:
        pass
    
# minmax[xaxis][negaxis], minmax[yaxis][negaxis] are minimum values;
# minmax[xaxis][posaxis], minmax[yaxis][posaxis] are maximum values.

grey = (0.7,0.7,0.7) # color of axes
tmajor = 10. # length of major tick marks in pixels
tminor = 5. # length of minor tick marks in pixels
border = 10. # border around graph
frac = 0.02 # fraction of range required before remaking axes
minorticks = 5 # number of minor tick intervals between major ticks
maxmajorticks = 3 # max number of major ticks (not including 0)
maxminorticks = (maxmajorticks+1)*minorticks # max number of minor ticks (4 between major ticks)
lastgdisplay = None # the most recently created gdisplay
gdotsize = 6.0 # diameter of gdot in pixels
dz = 0.01 # offset for plots relative to axes and labels
xaxis = 0
yaxis = 1
negaxis = 0
posaxis = 1
graphfont = "sans"
fontheight = 13 # font point size
charwidth = 9 # approximate character width
znormal = [0,0,1] # for faces
logticks = []
for i in range(4):
    logticks.append(log10(2*i+2)) # for displaying minor tick marks with log graphs
     
def loglabelnum(x): # determine what log labels to show, in what format
    number = abs(int(x))
    if number <= 1.01:
        marks = [1]
    elif number <= 2.01:
        marks = [1, 2]
    elif number <= 3.01:
        marks = [1, 2, 3]
    else:
        if not (number % 3): # is divisible by 3
            marks = [int(number/3), int((2*number/3)), int(number)]
        elif not (number % 2): # is divisible by 2
            marks = [int(number/2), int(number)]
        else:
            marks = [int((number+1)/2), int(number+1)]
    form = '1E{0}'
    return marks, form
     
def labelnum(x, loglabel): # determine what labels to show, in what format
    if loglabel:
        return loglabelnum(x)
    mantissa, exponent = modf(log10(x))
    number = 10**mantissa
    if number < 1:
        number = 10*number
        exponent = exponent-1
    if number >= 7.49: 
        number = 7.5
        marks = [2.5, 5.0, 7.5]
        extra = 1
    elif number >= 4.99: 
        number = 5
        marks = [2.5, 5.0]
        extra = 1
    elif number >= 3.99:
        number = 4
        marks = [2.0, 4.0]
        extra = 0
    elif number >= 2.99:
        number = 3
        marks = [1.0, 2.0, 3.0]
        extra = 0
    elif number >= 1.99:
        number = 3
        marks = [1.0, 2.0]
        extra = 0
    elif number >= 1.49:
        number = 1.5
        marks = [0.5, 1.0, 1.5]
        extra = 1
    else: 
        number = 1
        marks = [0.5, 1.0]
        extra = 1
    if exponent > 0:
        digits = 0
    else:
        digits = int(-exponent)+extra
    if digits < 3 and exponent <= 3:
        form = '{0:0.'+'{0:s}'.format(str(digits))+'f}'
    else:
        form = '{0:0.1E}'
    return (array(marks)*10**exponent).tolist(), form

def cleaneformat(string): # convert 2.5E-006 to 2.5E-6; 2.5E+006 to 2.5E6
    index = string.find('E') 
    if index == -1: return string # not E format
    index = index+1
    if string[index] == '-':
        index = index+1
    elif string[index] == '+':
        string = string[:index]+string[index+1:]
    while index < len(string) and string[index] == '0':
        string = string[:index]+string[index+1:]
    if string[-1] == '-':
        string = string[:-1]
    if string[-1] == 'E':
        string = string[:-1]
    return string

class gdisplay:
    def __init__(self, window=None, x=0, y=0, width=800, height=400,
                 title=None, xtitle=None, ytitle=None,
                 xmax=None, xmin=None, ymax=None, ymin=None,
                 logx=False, logy=False, visible=True,
                 foreground=None, background=None):
        global lastgdisplay
        lastgdisplay = self
        currentdisplay = display.get_selected()
        if title is None:
            title = 'Graph'
        self.width = width
        self.height = height
        if foreground is not None:
            self.foreground = foreground
        else:
            self.foreground = color.white
        if background is not None:
            self.background = background
        else:
            self.background = color.black
        self.visible = visible
        self.active = False
        if window:
            self.display = display(window=window,title=title, x=x, y=y,
                                   width=self.width, height=self.height,
                                   foreground=self.foreground, background=self.background,
                                   fov=0.01, userspin=False, uniform=False, autoscale=False,
                                   lights=[], ambient=color.gray(0))
        else:
            self.display = display(title=title, x=x, y=y, visible=self.visible,
                                   width=self.width, height=self.height,
                                   foreground=self.foreground, background=self.background,
                                   fov=0.01, userspin=False, uniform=False, autoscale=False,
                                   lights=[], ambient=color.gray(0))
        distant_light(direction=(0,0,1), color=color.white)
        self.autoscale = [1, 1]
        self.logx = logx
        self.logy = logy
        self.xtitle = xtitle
        self.ytitle = ytitle
        self.Lxtitle = label(display=self.display, visible=False, text="",
                             font=graphfont, height=fontheight, border=2,
                             xoffset=tminor, opacity=0, box=False, line=False)
        self.Lytitle = label(display=self.display, visible=False, text="",
                             font=graphfont, height=fontheight, border=2,
                             xoffset=tminor, opacity=0, box=False, line=False)
        if xtitle is not None: self.Lxtitle.text = xtitle
        self.xtitlewidth = len(self.Lxtitle.text)*charwidth
        if ytitle is not None: self.Lytitle.text = ytitle
        self.ytitlewidth = len(self.Lytitle.text)*charwidth
        self.mousepos = None
        self.showxy = label(display=self.display, color=self.foreground,
                            background=self.background,
                            xoffset=10, yoffset=8, border=0,
                            opacity=0, box=False, line=False, visible=False)
        gray = color.gray(0.5)
        self.horline = curve(display=self.display, color=gray, visible=False)
        self.vertline = curve(display=self.display, color=gray, visible=False)
        # For all axis-related quantities: [x axis 0 or y axis 1][neg axis 0 or pos axis 1]

        zerotextx = zerotexty = '0'
        if self.logx:
            zerotextx = '1'
        if self.logy:
            zerotexty = '1'
        self.zero = [label(display=self.display, pos=(0,0,0), text=zerotextx,
                    color=self.foreground, visible=False,
                    font=graphfont, height=fontheight, border=0,
                    yoffset=-tmajor, linecolor=grey, box=0, opacity=0),
                     
                    label(display=self.display, pos=(0,0,0), text=zerotexty,
                    color=self.foreground, visible=False,
                    font=graphfont, height=fontheight, border=2,
                    xoffset=-tmajor, linecolor=grey, box=0, opacity=0)]
        
        self.axis = [[None, None], [None, None]]
        self.makeaxis = [[True, True], [True, True]]
        self.lastlabel = [[0., 0.], [0., 0.]]
        self.format = [None, None]
        self.majormarks = [[None, None], [None, None]]
        self.lastminmax = [[0., 0.], [0., 0.]]
        self.minmax = [[0., 0.], [0., 0.]] # [x or y][negative 0 or positive 1]
        
        if self.logx:
            if xmax is not None:
                if xmax <= 0:
                    raise AttributeError("For a log scale, xmax must greater than zero")
                else: xmax = log10(float(xmax))
            if xmin is not None:
                if xmin <= 0:
                    raise AttributeError("For a log scale, xmin must be greater than zero")
                else: xmin = log10(float(xmin))
        if self.logy:
            if ymax is not None:
                if ymax <= 0:
                    raise AttributeError("For a log scale, ymax must greater than zero")
                else: ymax = log10(float(ymax))
            if ymin is not None:
                if ymin <= 0:
                    raise AttributeError("For a log scale, ymin must be greater than zero")
                else: ymin = log10(float(ymin))
                
        x0 = y0 = 0
        if xmax is not None:
            self.minmax[xaxis][posaxis] = xmax
            self.autoscale[xaxis] = False
            if xmax < 0:
                marks, form = labelnum(-xmax, self.logx)
                self.zero[xaxis].text = cleaneformat(form.format(xmax))
                self.makeaxis[xaxis][posaxis] = False
                if (xmin is None):
                    raise AttributeError("xmin must be specified to be less than xmax")
                x0 = xmax
            if (xmin is not None) and (xmin >= xmax):
                raise AttributeError("xmax must be greater than xmin")
        if xmin is not None:
            self.minmax[xaxis][negaxis] = xmin
            self.autoscale[xaxis] = False
            if xmin > 0:
                marks, form = labelnum(xmin, self.logx)
                self.zero[xaxis].text = cleaneformat(form.format(xmin))
                self.makeaxis[xaxis][negaxis] = False
                if (xmax is None):
                    raise AttributeError("xmax must be specified to be greater than xmin")
                x0 = xmin
            if (xmax is not None) and (xmin >= xmax):
                raise AttributeError("xmax must be greater than xmin")
        if ymax is not None:
            self.minmax[yaxis][posaxis] = ymax
            self.autoscale[yaxis] = False
            if ymax < 0:
                marks, form = labelnum(-ymax, self.logy)
                self.zero[yaxis].text = cleaneformat(form.format(ymax))
                self.makeaxis[yaxis][posaxis] = False
                if (ymin is None):
                    raise AttributeError("ymin must be specified to be less than ymax")
                y0 = ymax
            if (ymin is not None) and (ymin >= ymax):
                raise AttributeError("ymax must be greater than ymin")
        if ymin is not None:
            self.minmax[yaxis][negaxis] = ymin
            self.autoscale[yaxis] = False
            if ymin > 0:
                marks, form = labelnum(ymin, self.logy)
                self.zero[yaxis].text = cleaneformat(form.format(ymin))
                self.makeaxis[yaxis][negaxis] = False
                if (ymax is None):
                    raise AttributeError("ymax must be specified to be greater than ymin")
                y0 = ymin
            if (ymax is not None) and (ymin >= ymax):
                raise AttributeError("ymax must be greater than ymin")

        self.zero[0].pos = (x0, y0, 0)
        self.zero[1].pos = (x0, y0, 0)
        self.display.range = 1e-300

        self.minorticks = [ [ [], [] ], [ [],[] ] ] # all the minor ticks we'll ever use
        for axis in range(2):
            for axissign in range(2):
                for nn in range(maxminorticks):
                    if axis == xaxis:
                        self.minorticks[axis][axissign].append(label(display=self.display, yoffset=-tminor,
                            font=graphfont, height=fontheight, border=0,
                            linecolor=grey, visible=False, box=False, opacity=0))
                    else:
                        self.minorticks[axis][axissign].append(label(display=self.display, xoffset=-tminor,
                            font=graphfont, height=fontheight, border=0,
                            linecolor=grey, visible=False, box=False, opacity=0))

        self.majorticks = [ [ [], [] ], [ [],[] ] ] # all the major ticks we'll ever use
        for axis in range(2):
            for axissign in range(2):
                for nn in range(maxmajorticks):
                    if axis == xaxis:
                        self.majorticks[axis][axissign].append(label(display=self.display, yoffset=-tmajor, 
                            font=graphfont, height=fontheight, border=0, 
                            linecolor=grey, visible=False, box=False, opacity=0))
                    else:
                        self.majorticks[axis][axissign].append(label(display=self.display, xoffset=-tmajor, 
                            font=graphfont, height=fontheight, border=2,
                            linecolor=grey, visible=False, box=False, opacity=0))

        currentdisplay.select()

    def __del__(self):
        self.visible = False
        try:
            gdisplays.remove(self)
        except:
            pass

    def mouse(self, evt):
        m = evt
        if m.press == 'left':
            self.mousepos = self.display.mouse.pos+vector(10,20,30) # so mousepos != newpos
            self.horline.visible = True
            self.vertline.visible = True
            self.showxy.visible = True
            self.display.cursor.visible = False
        elif m.release == 'left':
            self.mousepos = None
            self.showxy.visible = False
            self.horline.visible = False
            self.vertline.visible = False
            self.display.cursor.visible = True
            
        newpos = self.display.mouse.pos
        if newpos != self.mousepos:
            self.mousepos = newpos
            xmax = self.display.range.x
            ymax = self.display.range.y
            xcenter = self.display.center.x
            ycenter = self.display.center.y
            self.horline.pos = [(xcenter-xmax,self.mousepos.y,.01),
                                (xcenter+xmax,self.mousepos.y,.01)]
            self.vertline.pos = [(self.mousepos.x,ycenter-ymax,.01),
                                 (self.mousepos.x,ycenter+ymax,.01)]
            v = self.showxy.pos = self.mousepos
            if self.logx: x = 10**v.x
            if self.logy: y = 10**v.y
            if v.x > xcenter:
                self.showxy.xoffset = -10
            else:
                self.showxy.xoffset = 10
            self.showxy.text = '({0:0.4g}, {1:0.4g})'.format(v.x,v.y)

    def setcenter(self):
        x0, y0 = self.getorigin()
        xright = self.minmax[xaxis][posaxis]
        xleft = self.minmax[xaxis][negaxis]
        ytop = self.minmax[yaxis][posaxis]
        ybottom = self.minmax[yaxis][negaxis]

        rightpixels = self.xtitlewidth
        leftpixels = 0
        if xleft == x0:
            leftpixels = 3*tmajor
        toppixels = bottompixels = 0
        if self.ytitlewidth:
            toppixels = 2*fontheight
        if ybottom == y0:
            bottompixels = tmajor+fontheight
            
        xrange = 0.55*(xright-xleft)*self.width/(self.width-(rightpixels+leftpixels))
        yrange = 0.55*(ytop-ybottom)*self.height/(self.height-(toppixels+bottompixels))
        xscale = xrange/(.5*self.width)
        yscale = yrange/(.5*self.height)
        x1 = xright+rightpixels*xscale
        x2 = x0+self.ytitlewidth*xscale
        if x2 > x1: # ytitle extends farther to the right than xaxis + xtitle
            xrange = 0.55*(x2-xleft)*self.width/(self.width-leftpixels)
            xscale = xrange/(.5*self.width)
            xright = x2
            rightpixels = 0
        if xrange == 0: xrange = 1e-300
        if yrange == 0: yrange = 1e-300
        self.display.range = (xrange,yrange,0.1)
        self.display.center = ((xright+xleft+(rightpixels-leftpixels)*xscale)/2.0,
                               (ytop+ybottom+(toppixels-bottompixels)*yscale)/2.0,0)

    def getorigin(self):
        return (self.zero[0].pos[0], self.zero[1].pos[1])
            
    def setminorticks(self, axis, axissign, loglabel, dmajor, dminor):
        ## For log axis labels, show the standard uneven log tick marks if dmajor == 1,
        ## but for dmajor > 1, show minor tick marks at the decade locations.
        ## Since we have only minorticks-1 = 4 minor tick marks between major tick marks,
        ## if dmajor > minorticks (=5), don't show any minor tick marks.
        if loglabel and (dmajor > minorticks): return 0
        x0,y0 = self.getorigin()
        limit = self.minmax[axis][axissign]
        if axis == xaxis: limit -= x0
        else: limit -= y0
        if axissign == negaxis:
            dminor = -dminor
        ntick = nmajor = nminor = 0
        exclude = minorticks
        if loglabel and (dmajor > 1):
            exclude = dmajor
            if dminor > 0:
                dminor = 1
            else:
                dminor = -1
        while True:
            ntick += 1
            tickpos = ntick*dminor
            if (ntick % exclude) == 0:
                nmajor += 1
                continue # no minor tick where there is a major one
            if loglabel: # have already excluded dmajor > minorticks (=5)
                if dmajor == 1:
                    if dminor > 0:
                        tickpos = dmajor*(nmajor+logticks[(ntick-1)%exclude])
                    else:
                        tickpos = dmajor*(-(nmajor+1)+logticks[3-((ntick-1)%exclude)])
            if dminor > 0:
                if tickpos > limit: break
            else:
                if tickpos < limit: break
            obj = self.minorticks[axis][axissign][nminor]
            if axis == xaxis:
                obj.pos = (x0+tickpos,y0,0)
            else:
                obj.pos = (x0,y0+tickpos,0)
            obj.visible = True
            nminor += 1
        return nminor
        
    def axisdisplay(self, axis, axissign):
        # axis = 0 for x axis, 1 for y axis
        # axissign = 0 for negative half-axis, 1 for positive half-axis
        if not self.makeaxis[axis][axissign]: return
        sign = 1
        if axissign == negaxis: sign = -1
        x0,y0 = self.getorigin()
        if axis == xaxis:
            loglabel = self.logx
        else:
            loglabel = self.logy
        if axis == xaxis: origin = x0
        else: origin = y0
        if self.axis[axis][axissign] is None: # new; no axis displayed up till now
            if self.minmax[axis][axissign] == origin: return # zero-length axis
            # Display axis and axis title
            if axis == xaxis:
                axispos = ([(x0,y0,0), (self.minmax[axis][axissign],y0,0)])
                titlepos = ([self.minmax[axis][posaxis],y0,0])
            else:
                axispos = ([(x0,y0,0), (x0,self.minmax[axis][axissign],0)])          
                titlepos = ([x0,self.minmax[axis][posaxis]+2*fontheight*self.display.range.y/(.5*self.width),0])
            self.axis[axis][axissign] = curve(pos=axispos, color=grey, display=self.display)
            if axis == xaxis and self.Lxtitle.text != "":
                self.Lxtitle.pos = titlepos
                self.Lxtitle.visible = True
            if axis == yaxis and self.Lytitle.text != "":
                self.Lytitle.pos = titlepos
                self.Lytitle.visible = True

            # Determine major tick marks and labels
            if origin != 0:
                newmajormarks, form = labelnum(self.minmax[axis][posaxis] -
                                               self.minmax[axis][negaxis], loglabel)
                dmajor = newmajormarks[0]
                for n, mark in enumerate(newmajormarks):
                    if origin > 0:
                        newmajormarks[n] += self.minmax[axis][negaxis]
                    else:
                        newmajormarks[n] -= self.minmax[axis][posaxis]
                    newmajormarks[n] = newmajormarks[n]
            else:
                if self.minmax[axis][posaxis] >= -self.minmax[axis][negaxis]:
                    newmajormarks, form = labelnum(self.minmax[axis][posaxis], loglabel)
                else:
                    if loglabel:
                        newmajormarks, form = labelnum(self.minmax[axis][negaxis], loglabel)
                    else:
                        newmajormarks, form = labelnum(-self.minmax[axis][negaxis], loglabel)
                dmajor = newmajormarks[0]
            self.format[axis] = form

            # Display major tick marks and labels
            nmajor = 0
            marks = []
            for x1 in newmajormarks:
                if x1 > abs(self.minmax[axis][axissign]): break # newmajormarks can refer to opposite half-axis
                if axissign == posaxis and x1 < origin: continue
                elif axissign == negaxis and x1 < abs(origin): continue
                marks.append(x1)
                obj = self.majorticks[axis][axissign][nmajor]
                if loglabel:
                    obj.text = self.format[axis].format(int(sign*x1))
                else:
                    obj.text = cleaneformat(self.format[axis].format(sign*x1))
                obj.color = self.foreground
                obj.visible = True
                if axis == xaxis:
                    obj.pos = [sign*x1,y0,0]
                    obj.yoffset = -tmajor 
                else:
                    obj.pos = [x0,sign*x1,0]
                    obj.xoffset = -tmajor
                nmajor = nmajor+1

            # Display minor tick marks
            self.setminorticks(axis, axissign, loglabel, dmajor, dmajor/minorticks)
                    
            if marks != []:
                self.majormarks[axis][axissign] = marks
                self.lastlabel[axis][axissign] = self.majormarks[axis][axissign][-1]
            else:
                self.lastlabel[axis][axissign] = 0

        else:
                    
            # Extend axis, which has grown
            if axis == xaxis:
                self.axis[axis][axissign].pos = [[x0,y0,0],[self.minmax[axis][axissign], y0, 0]]
            else:
                self.axis[axis][axissign].pos = [[x0,y0,0],[x0,self.minmax[axis][axissign],0]]
                
            # Reposition xtitle (at right) or ytitle (at top)
            if axis == xaxis and axissign == posaxis:
                self.Lxtitle.pos = (self.minmax[axis][posaxis],y0,0)
            if axis == yaxis and axissign == posaxis:
                self.Lytitle.pos = ([x0,self.minmax[axis][posaxis]+2*fontheight*self.display.range.y/(.5*self.width),0])
                
            # See how many majormarks are now needed, and in what format
            if self.minmax[axis][posaxis] >= -self.minmax[axis][negaxis]:
                newmajormarks, form = labelnum(self.minmax[axis][posaxis], loglabel)
            else:
                newmajormarks, form = labelnum(-self.minmax[axis][negaxis], loglabel)

            if (self.majormarks[axis][axissign] is not None) and (len(self.majormarks[axis][axissign]) > 0):
                # this axis already has major tick marks/labels
                olddmajor = self.majormarks[axis][axissign][0]
            else:
                olddmajor = 0.
            olddminor = olddmajor/minorticks
            dmajor = newmajormarks[0]
            dminor = dmajor/minorticks

            newformat = (form != self.format[axis])
            self.format[axis] = form
            check = (self.minmax[axis][axissign] >= self.lastlabel[axis][axissign]+dminor)
            if axissign == negaxis:
                check = (self.minmax[axis][axissign] <= self.lastlabel[axis][axissign]-dminor)
            needminor = check or (dminor != olddminor)
            needmajor = ((self.majormarks[axis][axissign] is None)
                        or (newmajormarks[-1] != self.majormarks[axis][axissign][-1]) or newformat)
                    
            if needmajor: # need new labels
                start = 0
                if (self.majormarks[axis][axissign] is None) or newformat or (dmajor != olddmajor):
                    marks = []
                else:
                    for num in newmajormarks:
                        if num > self.majormarks[axis][axissign][-1]:
                            start = num
                            break
                    marks = self.majormarks[axis][axissign]
                for nmajor in range(maxmajorticks):
                    obj = self.majorticks[axis][axissign][nmajor]
                    if nmajor < len(newmajormarks):
                        x1 = newmajormarks[nmajor]
                        if abs(self.minmax[axis][axissign]) >= x1:
                            if x1 < start:
                                continue
                        else:
                            obj.visible = False
                            continue
                    else:
                        obj.visible = False
                        continue
                    marks.append(x1)
                    if loglabel:
                        obj.text = self.format[axis].format(int(sign*x1))
                    else:
                        obj.text = cleaneformat(self.format[axis].format(sign*x1))
                    obj.color = self.foreground
                    obj.visible = True
                    if axis == xaxis:
                        obj.pos = [sign*x1,y0,0]
                    else:
                        obj.pos = [x0,sign*x1,0]

                if marks != []:
                    self.majormarks[axis][axissign] = marks
                        
            if needminor: # adjust minor tick marks
                nminor = self.setminorticks(axis, axissign, loglabel, dmajor, dminor)
                
                while nminor < maxminorticks:
                    self.minorticks[axis][axissign][nminor].visible = False
                    nminor = nminor+1

            self.lastlabel[axis][axissign] = dminor*int(self.minmax[axis][axissign]/dminor)
                   
    def resize(self, x, y):
        redox = redoy = False
        if self.autoscale[xaxis]:
            if x > self.lastminmax[xaxis][posaxis]:
                self.minmax[xaxis][posaxis] = x+frac*self.display.range[0]
                if (self.lastminmax[xaxis][posaxis] == 0 or
                        (self.minmax[xaxis][posaxis] >= self.lastminmax[xaxis][posaxis])):
                    redox = True
            elif x < self.lastminmax[xaxis][negaxis]:
                self.minmax[xaxis][negaxis] = x-frac*self.display.range[0]
                if (self.lastminmax[xaxis][negaxis] == 0 or
                        (self.minmax[xaxis][negaxis] <= self.lastminmax[xaxis][negaxis])):
                    redox = True
        elif not self.active:
            redox = redoy = True
                    
        if self.autoscale[yaxis]:
            if y > self.lastminmax[yaxis][posaxis]:
                self.minmax[yaxis][posaxis] = y+frac*self.display.range[1]
                if (self.lastminmax[yaxis][posaxis] == 0 or
                        (self.minmax[yaxis][posaxis] >= self.lastminmax[yaxis][posaxis])):
                    redoy = True
            elif y < self.lastminmax[yaxis][negaxis]:
                self.minmax[yaxis][negaxis] = y-frac*self.display.range[1]
                if (self.lastminmax[yaxis][negaxis] == 0 or
                        (self.minmax[yaxis][negaxis] <= self.lastminmax[yaxis][negaxis])):
                    redoy = True
        elif not self.active:
            redox = redoy = True

        if (redox or redoy ):
            self.setcenter() # approximate
            if redox:
                self.axisdisplay(xaxis,posaxis)
                self.lastminmax[xaxis][posaxis] = self.minmax[xaxis][posaxis]
                self.axisdisplay(xaxis,negaxis)
                self.lastminmax[xaxis][negaxis] = self.minmax[xaxis][negaxis]
            if redoy:
                self.axisdisplay(yaxis,posaxis)
                self.lastminmax[yaxis][posaxis] = self.minmax[yaxis][posaxis]
                self.axisdisplay(yaxis,negaxis)
                self.lastminmax[yaxis][negaxis] = self.minmax[yaxis][negaxis]
            self.setcenter() # revised

        if not self.active:
            self.active = True
            gdisplays.append(self)
            self.display.bind("mousedown mousemove mouseup", checkGraphMouse, self)
            self.zero[xaxis].visible = True
            self.zero[yaxis].visible = True
                    
def getgdisplay():
    return gdisplay()

def constructorargs(obj,arguments):
    if 'gdisplay' in arguments:
        obj.gdisplay = arguments['gdisplay']
    else:
        if lastgdisplay is None:
            obj.gdisplay = getgdisplay()
        else:
            obj.gdisplay = lastgdisplay
    if 'color' in arguments:
        obj.color = arguments['color']
    else:
        obj.color = obj.gdisplay.foreground
    if 'pos' in arguments:
        pos = array(arguments['pos'], float)
        if pos.ndim == 1: pos = array([pos])
        if obj.gdisplay.logx or obj.gdisplay.logy:
            pos = logarray(obj, pos)
        return(pos)
    else:
        return None

def primitiveargs(obj,arguments):
        if 'color' in arguments:
            c = arguments['color']
        else:
            c = obj.color
        if 'pos' in arguments:
            pos = array(arguments['pos'], float)
            if pos.ndim == 1: pos = array([pos])
            if obj.gdisplay.logx or obj.gdisplay.logy:
                pos = logarray(obj, pos)
            return (pos,c)
        else:
            raise RuntimeError("Cannot plot without specifying pos.")

def logarray(obj, pos):
    for p in pos:
        try:
            if obj.gdisplay.logx: p[0] = log10(p[0])
            if obj.gdisplay.logy: p[1] = log10(p[1])
        except:
            raise ValueError("Cannot take log of zero or a negative number: (x,y) = ({0:0.3G},{1:0.3G})".format(float(p[0]),float(p[1])))
    return pos

class gcurve:
    def __init__(self, dot=False, dot_color=None, size=8, shape='round', **args):
        pos = constructorargs(self,args)
        self.dot = dot
        self.dot_color = self.color
        if dot_color is not None:
            self.dot_color = dot_color
        self.size = size
        self.shape = shape
        if self.dot:
            self.dotobj = points(display=self.gdisplay.display,
                    color=self.dot_color, shape=self.shape, size=self.size)
        self.gcurve = curve(display=self.gdisplay.display, color=self.color)
        if pos is not None:
            self.plotpos(pos, self.color)

    def plot(self, **args):
        pos,c = primitiveargs(self,args)
        self.plotpos(pos, c)

    def plotpos(self, pos, c):
        for p in pos:
            self.gdisplay.resize(p[0],p[1])
            self.gcurve.append(pos=(p[0],p[1],2*dz), color=c)
        if self.dot:
            self.dotobj.pos = (pos[-1][0],pos[-1][1],3*dz)

class gdots:
    def __init__(self, size=5, shape='round', **args):
        pos = constructorargs(self,args)
        self.size = size
        self.shape = shape
        # For now, restrict to single color
        self.dots = points(display=self.gdisplay.display, color=self.color,
                               size=self.size, shape=self.shape)
        if pos is not None:
            self.plotpos(pos, self.color)
        else:
            if type(self.color) is ndarray and c.shape[0] > 1:
                raise RuntimeError("Cannot specify an array of colors without specifying pos.")

    def plot(self, **args):
        pos,c = primitiveargs(self,args)
        self.plotpos(pos, c)

    def plotpos(self, pos, c):
        for p in pos:
            self.gdisplay.resize(p[0],p[1])
            self.dots.append(pos=(p[0],p[1],2*dz), color=c)

class gvbars:
    def __init__(self, delta=1.0, **args):
        pos = constructorargs(self,args)
        self.delta = delta
        self.vbars = faces(display=self.gdisplay.display,
                           pos=[(0,0,0),(0,0,0),(0,0,0)],
                           normal=znormal, color=self.color)
        if pos is not None:
            self.plotpos(pos, self.color)

    def makevbar(self, pos):
        x,y = pos[0],pos[1]
        if y < 0.0:
            ymin = y
            ymax = 0
        else:
            ymin = 0.0
            ymax = y
        d = self.delta/2.0
        return [(x-d,ymin,dz),(x+d,ymin,dz),(x-d,ymax,dz),
                (x-d,ymax,dz),(x+d,ymin,dz),(x+d,ymax,dz)]

    def plot(self, **args):
        pos,c = primitiveargs(self,args)
        self.plotpos(pos, c)

    def plotpos(self, pos, c):
        for p in pos:
            self.gdisplay.resize(p[0],p[1]) 
            for pt in self.makevbar(p):
                self.vbars.append(pos=(pt[0],pt[1],dz), normal=znormal, color=c)

class ghbars:
    def __init__(self, delta=1.0, **args):
        pos = constructorargs(self,args)
        self.delta = delta
        self.hbars = faces(display=self.gdisplay.display,
                           pos=[(0,0,0),(0,0,0),(0,0,0)],
                           normal=znormal, color=self.color)
        if pos is not None:
            self.plotpos(pos, self.color)

    def makehbar(self, pos):
        x,y = pos[0],pos[1]
        if x < 0.0:
            xmin = x
            xmax = 0
        else:
            xmin = 0.0
            xmax = x
        d = self.delta/2.0
        return [(xmin,y-d,dz),(xmax,y-d,dz),(xmin,y+d,dz),
                (xmin,y+d,dz),(xmax,y-d,dz),(xmax,y+d,dz)]

    def plot(self, **args):
        pos,c = primitiveargs(self,args)
        self.plotpos(pos, c)

    def plotpos(self, pos, c):
        for p in pos:
            self.gdisplay.resize(p[0],p[1]) 
            for pt in self.makehbar(p):
                self.hbars.append(pos=(pt[0],pt[1],dz), normal=znormal, color=c)

class ghistogram:
    def __init__(self, bins=None, accumulate=0, average=0,
                 delta=None, gdisplay=None, color=None):
        if gdisplay is None:
            if lastgdisplay is None:
                gdisplay = getgdisplay()
            else:
                gdisplay = lastgdisplay
        if gdisplay.logx or gdisplay.logy:
            raise ValueError("A histogram cannot be logarithmic.")
        self.gdisplay = gdisplay
        self.bins = bins
        self.nhist = 0 # number of calls to plot routine
        self.accumulate = accumulate # add successive data sets
        self.average = average # display accumulated histogram divided by self.nhist
        if delta is None:
            self.delta = (bins[1]-bins[0])
        else:
            self.delta = delta
        if color is None:
            self.color = self.gdisplay.foreground
        else:
            self.color = color
        self.histaccum = zeros(len(bins))
        self.vbars = faces(display=self.gdisplay.display,
                           pos=[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)],
                           normal=znormal, color=self.color)
        self.gdisplay.resize(bins[0]-self.delta,1.0)
        self.gdisplay.resize(bins[-1]+self.delta,1.0)

    def makehistovbar(self, pos):
        x,y = pos[0],pos[1]
        d = 0.8*self.delta/2.0
        return [(x-d,0.0,dz),(x+d,0.0,dz),(x-d,y,dz),
                (x-d,y,dz),(x+d,0.0,dz),(x+d,y,dz)]

    def plot(self, data=None, accumulate=None, average=None, color=None):
        if color is None:
            color = self.color
        if accumulate is not None:
            self.accumulate = accumulate
        if average is not None:
            self.average = average
        if data is None: return
        n = searchsorted(sort(data), self.bins)
        n = concatenate([n, [len(data)]])
        histo = n[1:]-n[:-1]
        if self.accumulate:
            self.histaccum = self.histaccum+histo
        else:
            self.histaccum = histo
        self.nhist = self.nhist+1.
        ymax = max(self.histaccum)
        if ymax == 0.: ymax == 1.
        self.gdisplay.resize(self.bins[-1],ymax)
        for nbin in range(len(self.bins)):
            pos = [self.bins[0]+(nbin+0.5)*self.delta, self.histaccum[nbin]]
            if self.nhist == 1.:
                for pt in self.makehistovbar(pos):
                    self.vbars.append(pos=pt, normal=znormal)
            else:
                if self.accumulate and self.average: 
                    pos[1] /= self.nhist
                # (nbin+1) because self.vbars was initialized with one dummy bar
                self.vbars.pos[6*(nbin+1)+2][1] = pos[1]
                self.vbars.pos[6*(nbin+1)+3][1] = pos[1]
                self.vbars.pos[6*(nbin+1)+5][1] = pos[1]

