# Code to complete the various primitive types
# Users should never import this module directly.  All of the public types and
# functions will be explicitly imported by __init__.py
from __future__ import division, print_function
from . import cvisual
from sys import version_info

def import_check(name):
    if name == "Polygon":
        # Check for the Polygon module needed by the text and extrusion objects:
        imported = False
        try:
            from Polygon import Polygon
            imported = True
        except ImportError:
            print("The Polygon module is not installed,\n   so the text and extrusion objects are unavailable.")
        return imported
    elif name == "ttfquery":
        # Check for the font-handling modules needed by the text object:
        imported = False
        try:
            from ttfquery import describe, glyphquery, glyph
            imported = True
        except ImportError:
            print("The ttfquery and/or FontTools modules are not installed,\n   so the text object is unavailable.")
        return imported
    return False # only Polygon and ttfquery are meaningful

if import_check("Polygon"):
    from Polygon import Polygon
    from . import shapes

if import_check("ttfquery"):
    from ttfquery import describe, glyphquery, glyph

from .cvisual import vector
from numpy import array, asarray, zeros, arange, int32, float64, sin, cos, fromstring, uint8
from . import crayola
color = crayola
from math import pi

import wx as _wx
_App2 = _wx.App()

_platInfo = _wx.PlatformInformation()
_plat = _platInfo.GetOperatingSystemFamilyName() # 'Windows', 'Macintosh'

NCHORDS = 20.0 # number of chords in one coil of a helix

trail_list = [] # list of objects that have trails

# Scenegraph management:
#   Renderable objects which become visible need to be added
#     to the scene graph using EITHER
#       - display.add_renderable() if not in a frame, OR
#       - frame.add_renderable()
#   Renderable objects which become invisible need to be removed using
#     the corresponding remove_renderable function.
#   If the display or frame of a visible object changes, it needs to
#     be removed and added.
# The py_renderable class encapsulates this logic, as well as
#   a fair amount of construction and attribute access common to
#   all renderables.

class py_renderable(object):
    def __init__(self, **keywords):
        _other = keywords.get("_other")
        if _other:
            del keywords["_other"]
            super(py_renderable,self).__init__(_other)
            self.__dict__ = dict(_other.__dict__)
            self.__display = _other.display
            self.__frame = _other.frame
            self.__visible = _other.visible
        else:
            super(py_renderable, self).__init__()
            self.__display = cvisual.display_kernel.get_selected()
            self.__frame = None
            self.__visible = True

        if 'display' in keywords:
            self.__display = keywords['display']
            del keywords['display']
        if 'visible' in keywords:
            self.__visible = keywords['visible']
            del keywords['visible']
        if 'frame' in keywords:
            self.__frame = keywords['frame']
            del keywords['frame']

        if not _other: self.init_defaults(keywords)

        self.process_init_args_from_keyword_dictionary( keywords )
 
        if self.__frame:
            if self.__frame.display != self.__display:
                raise ValueError("Cannot initialize an object with a frame on a different display.")

        self.check_init_invariants()

        if self.__visible:
            if self.__frame:
                self.__frame.add_renderable(self)
            elif self.__display:
                self.__display.add_renderable(self)

    def __copy__( self, **keywords):
        return self.__class__(_other=self, **keywords)

    def check_init_invariants(self):
        pass

    def set_display(self, display):
        "For internal use only. The setter for the display property."
        if display != self.__display:
        # Check that we aren't screwing up a frame.
            if self.__frame:
                raise ValueError("""Cannot change displays when within a
                    frame.  Make frame None, first.""")
            if self.__display:
                self.__display.remove_renderable(self)
            self.__display = display
            self.__display.add_renderable(self)

    def get_display(self):
        "For internal use only.  The getter for the display property."
        return self.__display

    display = property( get_display, set_display)

    def get_frame(self):
        "For internal use only.  The getter for the frame property."
        return self.__frame

   # Overridden by the frame class below to add extra checks.
    def set_frame(self, frame):
        "For internal use only.  The setter for the frame property."
        if frame != self.__frame:
            if frame and (frame.display != self.__display):
                raise ValueError("Cannot set to a frame on a different display.")
            if frame and self.__frame:
                # Simply moving from one frame to another.
                self.__frame.remove_renderable(self)
                frame.add_renderable(self)
            elif frame and not self.__frame:
                # Moving into a reference frame when otherwise not in one.
                if self.__display:
                    self.__display.remove_renderable(self)
                frame.add_renderable(self)
            elif not frame and self.__frame:
                # Removing from a reference frame.
                self.__frame.remove_renderable(self)
                if self.__display:
                    self.__display.add_renderable(self)
            self.__frame = frame
            
    frame = property( get_frame, set_frame)

    def get_visible(self):
        "For internal use only.  The getter for the visible property."
        return self.__visible

    def set_visible(self, visible):
        "For internal use only.  The setter for the visible property."
        if visible and not self.__visible:
            if self.__frame:
                self.__frame.add_renderable(self)
            elif self.__display:
                self.__display.add_renderable(self)
        if not visible and self.__visible:
            if self.__frame:
                self.__frame.remove_renderable(self)
            elif self.__display:
                self.__display.remove_renderable(self)
        self.__visible = visible

    visible = property( get_visible, set_visible)

    def init_defaults(self, keywords):
        self.color = self.display.foreground
        if isinstance(self, cvisual.light):
            self.color = (1,1,1)
        elif 'material' not in keywords:
            self.material = self.display.material

    def process_init_args_from_keyword_dictionary( self, keywords ):
        self.primitive_object = self
        if 'axis' in keywords: #< Should be set before 'length'
            self.axis = keywords['axis']
            del keywords['axis']
        if 'color' in keywords: 
            self.color = keywords['color']
            del keywords['color']
        if 'trail_type' in keywords:
            self.trail_type = keywords['trail_type']
            del keywords['trail_type']
            if not (self.trail_type == 'curve' or self.trail_type == 'points'):
                raise RuntimeError("trail_type must be 'curve' or 'points', not '"+trail_type+"'")
        else:
            self.trail_type = 'curve'
        self.interval = 1
        if 'interval' in keywords:
            self.interval = keywords['interval']
            del keywords['interval']
        if 'make_trail' in keywords:
            make_trail = keywords['make_trail']
            del keywords['make_trail']
            if self.trail_type == 'curve':
                self.trail_object = curve(frame=self.__frame, color=self.color)
            else:
                self.trail_object = points(frame=self.__frame, color=self.color)
            if make_trail and self.interval > 0:
                if 'pos' in keywords:
                    self.pos = keywords['pos']
                    del keywords['pos']
                    self.trail_object.pos = self.pos
            self.retain = -1
            self.interval_count = 0
            self.updated = False # True if trail_update has been called
            trail_list.append(self)
        else:
            make_trail = None

        # Assign all other properties
        for key, value in keywords.items():
            setattr(self, key, value)

        if not (make_trail is None):
            self.make_trail = make_trail

def trail_update(obj):
    # trail_update does not detect changes such as ball.pos.x += 1
    # which are detected in create_display/_Interact which looks at trail_list
    if obj.interval == 0: return
    obj.updated = True
    obj.interval_count += 1
    if len(obj.trail_object.pos) == 0:
        obj.trail_object.append(pos=obj.pos)
        obj.interval_count -= 1
    if obj.interval_count == obj.interval:
        if obj.pos != obj.trail_object.pos[-1]:
            obj.trail_object.append(pos=obj.pos, retain=obj.retain)
        obj.interval_count = 0

class py_renderable_uniform (py_renderable):
    def check_init_invariants(self):
        if not self.display.uniform:
            raise RuntimeError("Do not create " + self.__class__.__name__ + " with nonuniform axes.")      

class py_renderable_arrayobject (py_renderable):
   # Array objects are likely to need special handling in various places

    def get_red( self):
        return self.color[:,0]
    def get_green(self):
        return self.color[:,1]
    def get_blue(self):
        return self.color[:,2]
    def get_x(self):
        return self.pos[:,0]
    def get_y(self):
        return self.pos[:,1]
    def get_z(self):
        return self.pos[:,2]

    # Also none of them support opacity yet.   
    def set_opacity(self, opacity):
        raise RuntimeError("Cannot yet specify opacity for curve, extrusion, faces, convex, or points.")
    opacity = property( None, set_opacity, None)

################################################################################
# Complete each type.

class distant_light (py_renderable, cvisual.distant_light):
    def set_pos(self, _): raise AttributeError("Attempt to set pos of a distant_light object.")
    pos = property(None,set_pos)

class local_light (py_renderable, cvisual.local_light):
    def set_direction(self, _): raise AttributeError("Attempt to set direction of a local_light object.")
    direction = property(None,set_direction)

class arrow (py_renderable_uniform, cvisual.arrow):
    pass

class cone (py_renderable_uniform, cvisual.cone):
    pass

class cylinder (py_renderable_uniform, cvisual.cylinder):
    pass

class sphere (py_renderable_uniform, cvisual.sphere):
    pass

class ring (py_renderable_uniform, cvisual.ring):
    pass

class box (py_renderable_uniform, cvisual.box):
    pass
            
class ellipsoid (py_renderable_uniform, cvisual.ellipsoid):
    pass

class pyramid (py_renderable_uniform, cvisual.pyramid ):
    pass

### Test routine for the text_to_bitmap function
### Not currently working due to changes to text_to_bitmap for label object
##bitmap = text_to_bitmap('VPython', height=36, color=color.red, background=color.green,
##                   font='sans', style='italic', weight='normal')
##T = materials.texture(data=bitmap, mipmap=False, mapping='sign')
##box(material=T, axis=(0,0,1))

def text_to_bitmap(text, color=(1,1,1), background=(0,0,0), opacity=1,
                   height=13, font='sans',
                   style='normal', weight='normal'):
    # Basic algorithm provided by Chris Barker in the wxPython forum
    
    # Later (2012 Dec. 21) Chris suggested the following approach, which might be worth trying:
    # a) Draw your text black on white.
    # b) When you add the alpha channel make the alpha value scale with 
    #    the "blackness" of the pixel.
    # c) Turn all non-white pixels to full black. 
    # d) If you want the text some other color, change the color at the last step
    
    if font == 'sans':
        wfont = _wx.FONTFAMILY_SWISS
        if _plat == 'Windows':
            fudge = 11.0  # fudge factor for backwards compatibility
        elif _plat == 'Macintosh':
            fudge = 15.0
        else: 
            fudge = 9.0 # Linux; need to check this
    elif font == 'serif':
        wfont = _wx.FONTFAMILY_ROMAN
        if _plat == 'Windows':
            fudge = 10.0  # fudge factor for backwards compatibility
        elif _plat == 'Macintosh':
            fudge = 14.0
        else: 
            fudge = 7.0 # Linux; need to check this
    elif font == 'monospace':
        wfont = _wx.FONTFAMILY_MODERN
        if _plat == 'Windows':
            fudge = 10.0  # fudge factor for backwards compatibility
        elif _plat == 'Macintosh':
            fudge = 14.0
        else: 
            fudge = 9.5 # Linux; need to check this
    else:
        raise ValueError("font should be 'serif', 'sans', or 'monospace'")

    fudge /= 13.0 # default height is 13
    
    if style == 'normal':
        wstyle = _wx.FONTSTYLE_NORMAL
    elif style == 'italic':
        wstyle = _wx.FONTSTYLE_ITALIC
    else:
        raise ValueError("font style should be 'normal' or 'italic'")

    if weight == 'normal':
        wweight = _wx.FONTWEIGHT_NORMAL
    elif weight == 'bold':
        wweight = _wx.FONTWEIGHT_BOLD
    else:
        raise ValueError("font weight should be 'normal' or 'bold'")

    dc = _wx.MemoryDC()
    height = int(fudge*height + 0.5)
    dc.SetFont(_wx.Font(height, wfont, wstyle, wweight))
    while text and text[0] == '\n': text = text[1:]
    while text and  text[-1] == '\n': text = text[:-1]
    lines = text.split('\n')
    maxwidth = 0
    totalheight = 0
    heights = []
    for line in lines:
        if line == '': line = ' '
        w,h = dc.GetTextExtent(line)
        h += 1
        w += 2
        if w > maxwidth: maxwidth = w
        heights.append(totalheight)
        totalheight += h
    if 'phoenix' in _wx.PlatformInfo:
        bmp = _wx.Bitmap(maxwidth,totalheight) # Phoenix
    else:
        bmp = _wx.EmptyBitmap(maxwidth,totalheight) # classic
    dc.SelectObject(bmp)
    
    fore = (int(255*color[0]),
            int(255*color[1]),
            int(255*color[2]))
    dc.SetTextForeground(fore)
    
    back = (int(255*background[0]),
            int(255*background[1]),
            int(255*background[2]))
    if (back == fore):
        if fore == (0,0,0): back = (255,255,255)
        elif fore == (255,255,255): back = (0,0,0)
        else: back = (fore[0]//2, fore[1]//2, fore[2]//2)
    brush = _wx.Brush(back)
    dc.SetBackground(brush)
    dc.Clear()

    for n, line in enumerate(lines):
        dc.DrawText(line, 1, heights[n])

    dc.SelectObject( _wx.NullBitmap )
    if 'phoenix' in _wx.PlatformInfo:
        img = bmp.ConvertToImage() # Phoenix
        data = asarray(img.GetData()) # Phoenix; maybe should be GetDataBuffer()
    else:
        img = _wx.ImageFromBitmap(bmp) # classic
        data = fromstring(img.GetData(), dtype=uint8) # classic
    return maxwidth, totalheight, back, data
        
def get_bitmap(obj):
    w, h, back, data = text_to_bitmap(obj.text, obj.color, obj.background, obj.opacity,
                        obj.height, obj.font, "normal", "normal")
    cvisual.label.set_bitmap(obj, data, w, h, back[0], back[1], back[2])

class label (py_renderable, cvisual.label):
    def init_defaults( self, keywords ):
        if 'linecolor' not in keywords:
            self.linecolor = self.display.foreground
        if 'background' not in keywords:
            self.background = self.display.background
        if 'font' not in keywords:
            self.font = 'sans'
        super(label, self).init_defaults( keywords )

class frame (py_renderable_uniform, cvisual.frame):
    def set_frame(self, frame):
        #Check to ensure that we are not establishing a cycle of reference frames.
        frame_iterator = frame
        while frame_iterator:
            if frame_iterator.frame is self:
                raise ValueError("Attempted to create a cycle of reference frames.")
            frame_iterator = frame_iterator.frame
        py_renderable_uniform.set_frame( self, frame)

class faces( py_renderable_arrayobject, cvisual.faces ):

    def set_pos(self, positions):
        # We come here only if self.pos = array(); we go elsewhere for self.pos[i:j] = array().
        # The following somewhat odd scheme stopped what had been a bad memory leak:
        A = array(positions)
        if len(self.pos) == A.shape[0]:
            self.pos[:A.shape[0]] = A
        else:
            cvisual.faces.set_pos(self, A)

    def set_normal(self, normals):
        # We come here only if self.normal = array(); we go elsewhere for self.normal[i:j] = array().
        # The following somewhat odd scheme stopped what had been a bad memory leak:
        A = array(normals)
        if len(self.normal) == A.shape[0]:
            self.normal[:A.shape[0]] = A
        else:
            cvisual.faces.set_normal(self, A)

    def set_color(self, colors):
        # We come here only if self.color = array(); we go elsewhere for self.color[i:j] = array().
        # The following somewhat odd scheme stopped what had been a bad memory leak:
        A = array(colors)
        if len(self.color) == A.shape[0]:
            self.color[:A.shape[0]] = A
        else:
            cvisual.faces.set_color(self, A)
    
    pos = property( cvisual.faces.get_pos, set_pos, None)
    normal = property( cvisual.faces.get_normal, set_normal, None)
    color = property( cvisual.faces.get_color, set_color, None)
    red = property( py_renderable_arrayobject.get_red, cvisual.faces.set_red, None)
    green = property( py_renderable_arrayobject.get_green, cvisual.faces.set_green, None)
    blue = property( py_renderable_arrayobject.get_blue, cvisual.faces.set_blue, None)

class curve (py_renderable_arrayobject, cvisual.curve ):

    def set_pos(self, positions):
        
        try:
            self.up = positions.up
            # Positions created by paths library (or equivalent)
            cvisual.curve.set_pos(self, array(positions.pos))
            return
        except:
            pass
        
        try: # A Polygon object (2D), or a simple list of positions
            pts = positions.contour(0)
        except:
            # We come here only if self.pos = array(); we go elsewhere for self.pos[i:j] = array().
            # The following somewhat odd scheme stopped what had been a bad memory leak:
            A = array(positions)
            if len(self.pos) > 0 and len(self.pos) == A.shape[0]:
                self.pos[:A.shape[0]] = A
            else:
                cvisual.curve.set_pos(self, A)
            return

        # A Polygon object (2D)
        if len(positions) > 1:
            raise ValueError("A Polygon used for pos must represent a single contour.")
        newpts = []
        for pt in pts:
            newpt = vector(pt[0],0,-pt[1])
            newpts.append(newpt)
        # Polygon produces closed curves but doesn't always make the last point
        # equal to the first point, so we must check:
        if newpts[-1] != newpts[0]:
            newpts.append(newpts[0])
        cvisual.curve.set_pos(self, array(newpts))

    def set_color(self, colors):
        # We come here only if self.color = array(); we go elsewhere for self.color[i:j] = array().
        # The following somewhat odd scheme stopped what had been a bad memory leak:
        A = array(colors)
        if len(self.color) == A.shape[0]:
            self.color[:A.shape[0]] = A
        else:
            cvisual.curve.set_color(self, A)
    
    pos = property( cvisual.curve.get_pos, set_pos, None)
    color = property( cvisual.curve.get_color, set_color, None)
    x = property( py_renderable_arrayobject.get_x, cvisual.curve.set_x, None)
    y = property( py_renderable_arrayobject.get_y, cvisual.curve.set_y, None)
    z = property( py_renderable_arrayobject.get_z, cvisual.curve.set_z, None)
    red = property( py_renderable_arrayobject.get_red, cvisual.curve.set_red, None)
    green = property( py_renderable_arrayobject.get_green, cvisual.curve.set_green, None)
    blue = property( py_renderable_arrayobject.get_blue, cvisual.curve.set_blue, None)

class extrusion (py_renderable_uniform, py_renderable_arrayobject, cvisual.extrusion ):

    def get_faces(self):
        current_visible = self.visible
        self.visible = False # prevent normal rendering while executing _faces_render()
        data = self._faces_render()
        self.visible = current_visible
        L = len(data)/3
        return (data[:L], data[L:2*L], data[2*L:])

    def create_faces(self, keep=False):
        currentvisible = self.visible
        if self.frame:
            fcurrentvisible = self.frame.visible
        (pos, normal, color) = self.get_faces()
        self.visible = False
        newfaces = faces(frame=self.frame, pos=pos, normal=normal, color=color,
                         material=self.material, visible=currentvisible,
                         display=self.display, up=self.up)
        if newfaces.frame:
            newfaces.frame.visible = fcurrentvisible
        if not keep:
            # Cannot effectively delete "self", so destroy this extrusion's attributes:
            self.pos = [(0,0,0),(0,0,0)]
            self.shape = [(0,0),(1,0),(0,0)]
            self.color = [(0,0,0),(0,0,0)]
            self.show_start_face = False
            self.show_end_face = False
            self.material = None
        return newfaces

    def xycurl(self, s): # returns positive if counterclockwise (seen from front)
        c = 0
        for i in range(len(s)-1):
            x1 = s[i][0]
            y1 = s[i][1]
            x2 = s[i+1][0]
            y2 = s[i+1][1]
            c += x1*(y2-y1) - y1*(x2-x1)
        return c

    def xzcurl(self, s): # returns positive if counterclockwise (seen from above)
        c = 0
        for i in range(len(s)-1):
            x1 = s[i][0]
            z1 = s[i][2]
            x2 = s[i+1][0]
            z2 = s[i+1][2]
            c += z1*(x2-x1) - x1*(z2-z1)
        return c
    
    def make_shapedata(self, datalist, closed):
        ncontours = len(datalist)
        pointers = zeros(shape=(ncontours+1,2),dtype=int32) # specify type in detail
        pointers[0] = (ncontours,closed)
        datalength = 0
        for i, c in enumerate(datalist):
            length = len(c)
            pointers[i+1]=(length,datalength)
            datalength += length
        contours = zeros(shape=(datalength,2),dtype=float64) # specify type in detail
        for i, c in enumerate(datalist):
            contours[pointers[i+1][1]:pointers[i+1][1]+pointers[i+1][0]] = c
        return contours, pointers

    def set_shape(self, shape):
        if not import_check("Polygon"):
            return
        # shape can be a Polygon object or a simple list of points
        # Construct a pointer array with this integer format:
        # (number of contours,closed), where closed=1 for shape=Polygon object
        #    but closed=0 for shape=list of points with final != initial.
        #    In both Polygon and list cases if final == initial, final is discarded
        # (length of 1st contour, offset in array to data for 1st contour)
        # (length of 2nd contour, offset in array to data for 2nd contour)
        # .....
        # Construct a contour array with this float format:
        # (x,y) data for 1st contour
        # (x,y) data for 2nd contour
        # .....
        datalist = []
        
        if isinstance(shape, str):
            shape = shapes.text(text=shape, align='center')
        else:
            try:
                shape.pos # could be a paths.xxx object, a list of vectors
                s = []
                for v in shape.pos:
                    s.append((v.x, -v.z))
                shape = Polygon(s)
            except:
                pass
            
        try:
            for i in range(len(shape)):
                s = list(shape.contour(i))
                if s[-1] == s[0]: # discard final point if same as initial
                    s = s[:-1]
                c = self.xycurl(s) # positive if CCW
                # Require hole contours to run CCW, external contours CW
                if (shape.isHole(i) and (c < 0) or
                    (not shape.isHole(i)) and (c > 0)):
                    # Need to reverse the order of points in this contour
                    s.reverse()
                datalist.append(s)
            closed = True
            isPolygon = True
        except:
            s = shape
            # Require (external) closed contour to run clockwise (curl > 0)
            if self.xycurl(s) > 0: # Need to reverse the order of points in this contour
                s.reverse()
            if s[-1] == s[0]: # discard final point if same as initial, mark closed
                s = s[:-1]
                closed = True
            else:
                closed = False
            isPolygon = False
            datalist = array([s], float)

        contours, pcontours = self.make_shapedata(datalist, closed)
        if isPolygon:
            strips, pstrips = self.make_shapedata(shape.triStrip(), closed)
        else:
            strips = array([(0,0)])
            pstrips = array([(0,0)])
        # Send pointers, contour data, and shape.triStrip data to Visual
        self.set_contours(contours, pcontours, strips, pstrips)

    def set_pos(self, positions):
        if not import_check("Polygon"):
            return
        
        try:
            self.up = positions.up
            # Positions created by paths library (or equivalent)
            cvisual.extrusion.set_pos(self, array(positions.pos))
            return
        except:
            pass
        
        try: # A Polygon object (2D), or a simple list of positions
            pts = positions.contour(0)
        except:
            cvisual.extrusion.set_pos(self, array(positions))
            return

        # A Polygon object (2D)
        if len(positions) > 1:
            raise ValueError("A Polygon used for pos must represent a single contour.")
        newpts = []
        for pt in pts:
            newpt = vector(pt[0],0,-pt[1])
            newpts.append(newpt)
        # Polygon produces closed curves but doesn't always make the last point
        # equal to the first point, so we must check:
        if newpts[-1] != newpts[0]:
            newpts.append(newpts[0])
        if self.xzcurl(newpts) < 0: # positive if CCW as seen from above
            newpts.reverse()
        cvisual.extrusion.set_pos(self, array(newpts))
        
    def get_xscale(self):
        return self.scale[:,0]
    def get_yscale(self):
        return self.scale[:,1]

    def get_shape(self):
        raise AttributeError("An extrusion's shape is not readable.")
        
    pos = property( cvisual.extrusion.get_pos, set_pos, None)
    color = property( cvisual.extrusion.get_color, cvisual.extrusion.set_color, None)
    x = property( py_renderable_arrayobject.get_x, cvisual.extrusion.set_x, None)
    y = property( py_renderable_arrayobject.get_y, cvisual.extrusion.set_y, None)
    z = property( py_renderable_arrayobject.get_z, cvisual.extrusion.set_z, None)
    red = property( py_renderable_arrayobject.get_red, cvisual.extrusion.set_red, None)
    green = property( py_renderable_arrayobject.get_green, cvisual.extrusion.set_green, None)
    blue = property( py_renderable_arrayobject.get_blue, cvisual.extrusion.set_blue, None)
    shape = property( get_shape, set_shape, None)
    scale = property( cvisual.extrusion.get_scale, cvisual.extrusion.set_scale, None)
    xscale = property( get_xscale, cvisual.extrusion.set_xscale, None)
    yscale = property( get_yscale, cvisual.extrusion.set_yscale, None)
    twist = property( cvisual.extrusion.get_twist, cvisual.extrusion.set_twist, None)
    
class points ( py_renderable_arrayobject, cvisual.points ):
    
    pos = property( cvisual.points.get_pos, cvisual.points.set_pos, None)
    color = property( cvisual.points.get_color, cvisual.points.set_color, None)
    x = property( py_renderable_arrayobject.get_x, cvisual.points.set_x, None)
    y = property( py_renderable_arrayobject.get_y, cvisual.points.set_y, None)
    z = property( py_renderable_arrayobject.get_z, cvisual.points.set_z, None)
    red = property( py_renderable_arrayobject.get_red, cvisual.points.set_red, None)
    green = property( py_renderable_arrayobject.get_green, cvisual.points.set_green, None)
    blue = property( py_renderable_arrayobject.get_blue, cvisual.points.set_blue, None)

class convex( py_renderable_arrayobject, py_renderable_uniform, cvisual.convex ):
    pos = property( cvisual.convex.get_pos, cvisual.convex.set_pos, None)

class helix(py_renderable_uniform, py_renderable):
    def __init__( self, _other=None, pos=vector(),
        x=None, y=None, z=None, red=None, green=None, blue=None,
        axis=vector(1,0,0), radius=1.0, length=None, up=vector(0,1,0),
        coils=5, thickness=None, color=color.white, visible=True, **keywords):
        if 'display' in keywords:
            disp = keywords['display']
            del keywords['display']
        else:
            disp = cvisual.display_kernel.get_selected()
        if (not disp.uniform):
           raise RuntimeError("Do not create helix with nonuniform axes.")
        if 'frame' in keywords:
            fr = keywords['frame']
            del keywords['frame']
        else:
            fr = None
        self.process_init_args_from_keyword_dictionary( keywords )
        if x is not None:
            pos[0] = x
        if y is not None:
            pos[1] = y
        if z is not None:
            pos[2] = z
        if red is not None:
            color[0] = red
        if green is not None:
            color[1] = green
        if blue is not None:
            color[2] = blue
        self.__color = color
        axis = vector(axis)
        if length is None:
            length = axis.mag
        self.__length = length
        self.__axis = axis
        self.__radius = radius
        self.__up = up
        self.__coils = coils
        self.__thickness = radius/20.
        if thickness:
            self.__thickness = thickness
        self.__frame = frame(display=disp, frame=fr, pos=pos, axis=axis.norm(), up=up)
        self.helix = curve( frame = self.__frame, radius = self.__thickness/2.,
            color = color)
        self.create_pos()
      
    def create_pos(self):
        k = self.coils*(2*pi/self.__length)
        dx = (self.length/self.coils)/NCHORDS
        x_col = arange(0, self.__length+dx, dx)
        pos_data = zeros((len(x_col),3), float64)
        pos_data[:,0] = arange(0, self.__length+dx, dx)
        pos_data[:,1] = (self.radius) * sin(k*pos_data[:,0])
        pos_data[:,2] = (self.radius) * cos(k*pos_data[:,0])
        self.helix.pos = pos_data

    def set_pos(self, pos):
        self.__frame.pos = vector(pos)
    def get_pos(self):
        return self.__frame.pos

    def set_x(self, x):
        self.__frame.pos.x = x
    def get_x(self):
        return self.__frame.pos.x

    def set_y(self, y):
        self.__frame.pos.y = y
    def get_y(self):
        return self.__frame.pos.y

    def set_z(self, z):
        self.__frame.pos.z = z
    def get_z(self):
        return self.__frame.pos.z

    def set_color(self, color):
        self.__color = self.helix.color = color
    def get_color(self):
        return self.__color

    def set_red(self, red):
        self.helix.red = red
    def get_red(self):
        return self.helix.red

    def set_green(self, green):
        self.helix.green = green
    def get_green(self):
        return self.helix.green

    def set_blue(self, blue):
        self.helix.blue = blue
    def get_blue(self):
        return self.helix.blue

    def set_radius(self, radius):
        scale = radius/self.__radius
        self.__radius = radius
        self.helix.y *= scale
        self.helix.z *= scale
    def get_radius(self):
        return self.__radius

    def set_axis(self, axis):
        axis = vector(axis)
        self.__axis = axis
        self.__frame.axis = axis.norm()
        self.set_length(axis.mag)
    def get_axis(self):
        return self.__axis

    def set_length(self, length):
        self.helix.x *= (length/self.__length)
        self.__length = length
        self.__frame.axis = self.__axis.norm()
        self.__axis = length*self.__frame.axis
    def get_length(self):
        return self.__length

    def set_coils(self, coils):
        if self.__coils == coils: return
        self.__coils = coils
        self.create_pos()
    def get_coils(self):
        return self.__coils

    def set_thickness(self, thickness):
        if self.__thickness == thickness: return
        self.__thickness = thickness
        self.helix.radius = thickness/2.
    def get_thickness(self):
        return self.__thickness

    def set_display(self, disp):
        self.helix.display = self.frame.display = disp
    def get_display(self):
        return self.helix.display

    def set_frame(self, fr):
        self.__frame.frame = fr
    def get_frame(self):
        return self.__frame.frame

    def set_up(self, up):
        self.__frame.up = up
    def get_up(self):
        return self.__frame.up

    def set_visible(self, visible):
        self.helix.visible = visible
    def get_visible(self):
        return self.helix.visible

    pos = property( get_pos, set_pos, None)
    x = property( get_x, set_x, None)
    y = property( get_y, set_y, None)
    z = property( get_z, set_z, None)
    color = property( get_color, set_color, None)
    red = property( get_red, set_red, None)
    green = property( get_green, set_green, None)
    blue = property( get_blue, set_blue, None)
    axis = property( get_axis, set_axis, None)
    radius = property( get_radius, set_radius, None)
    coils = property( get_coils, set_coils, None)
    thickness = property( get_thickness, set_thickness, None)
    length = property( get_length, set_length, None)
    display = property( get_display, set_display, None)
    frame = property( get_frame, set_frame, None)
    up = property( get_up, set_up, None)
    visible = property( get_visible, set_visible, None)

class text(object):

    def __init__(self, pos=(0,0,0), axis=(1,0,0), up=(0,1,0), height=1, color=color.white, 
                 x=None, y=None, z=None, red=None, green=None, blue=None,
                 text="", font="serif", align='left', visible=True, twosided=True,
                 depth=0.2, material=None, spacing=0.03, vertical_spacing=None, **keywords):
        if not import_check("ttfquery"):
            return
        self.initial = True
        self.__text = text
        if 'display' in keywords:
            self.__display = keywords['display']
            del keywords['display']
        else:
            self.__display = cvisual.display_kernel.get_selected()
        if (not self.display.uniform):
           raise RuntimeError("Do not create 3D text with nonuniform axes.")
        if 'frame' in keywords:
            fr = keywords['frame']
            del keywords['frame']
        else:
            fr = None

        if x is not None:
            pos[0] = x
        if y is not None:
            pos[1] = y
        if z is not None:
            pos[2] = z
        if red is not None:
            color[0] = red
        if green is not None:
            color[1] = green
        if blue is not None:
            color[2] = blue
        self.__color = color
        self.__material = material
        self.__visible = visible
        self.__up = up
        self.__twosided = twosided
        self.__font = font
        self.__height = float(height)
        self.__spacing = spacing
        self.__vertical_spacing = vertical_spacing
        try:
            if len(depth) == 3:
                self.__depth = vector(depth)
        except:
            self.__depth = vector(0,0,depth)
        self.__frame = frame(display=self.__display, frame=fr, pos=pos, axis=vector(axis).norm(), up=up)
        self.check_align(align)
        self.paintText()
        self.initial = False

    def get_faces(self):
        current_visible = self.visible
        self.visible = False # prevent normal rendering while executing _faces_render()
        data = self.extrusion._faces_render()
        self.visible = current_visible
        L = len(data)/3
        return (data[:L], data[L:2*L], data[2*L:])

    def create_faces(self, keep=False):
        currentvisible = self.visible
        if self.frame:
            fcurrentvisible = self.frame.visible
        (pos, normal, color) = self.extrusion.get_faces()
        self.visible = False
        newfaces = faces(frame=self.frame, pos=pos, normal=normal, color=color,
                         material=self.material, visible=currentvisible,
                         display=self.display, up=self.up)
        if newfaces.frame:
            newfaces.frame.visible = fcurrentvisible
        if not keep:
            # Cannot effectively delete "self", so destroy this extrusion's attributes:
            self.text = ""
            self.extrusion.pos = [(0,0,0),(0,0,0)]
            self.extrusion.shape = [(0,0),(1,0),(0,0)]
            self.extrusion.color = [(0,0,0),(0,0,0)]
            self.extrusion.show_start_face = False
            self.extrusion.show_end_face = False
        return newfaces

    def fontlist(self):
        return shapes.findSystemFonts()
        
    def destroyText(self):
        try:
            self.extrusion.visible = False
            del self.extrusion
        except:
            pass

    def paintText(self):
        # For text(text="The quick brown dog jumps over the lazy dog"):
        # 6 ms render time using extrusion for text
        # 8 ms render time the old way, not using extrusion
        # Also, the cycle times are much longer without using extrusion.
        # I don't understand this, since when using extrusion the
        # faces have to be constructed for each rendering, whereas
        # the old text object made the faces just once, and then the
        # (three) faces objects were rendered (front, back, sides).
        
        if not self.initial:
            self.destroyText()
               
        shape = shapes.text(text=self.__text, font=self.__font,
                    align=self.__align, height=self.__height, info=True,
                    spacing=self.__spacing, vertical_spacing=self.__vertical_spacing)

        dy = self.__height-(shape.upper_left.y-shape.lower_left.y)/2
        if self.__align == 'left':
            self.displace = vector(shape.width/2,dy,0)
            dstart = vector(0,dy,0)
        elif self.__align == 'right':
            self.displace = vector(-shape.width/2,dy,0)
            dstart = vector(0,dy,0)
        else:
            self.displace = vector(0,dy,0)
            dstart = vector(0,dy,0)
        self.__start = shape.start+dstart
        ystart = self.__start.y
        for line in range(len(shape.starts)):
            shape.starts[line] = vector(shape.starts[line].x,ystart-line*shape.vertical_spacing,0)
        self.__starts = shape.starts

        self.__width = shape.width
        self.__widths = shape.widths
        self.__descent = shape.descent
        self.__vertical_spacing = shape.vertical_spacing
        self.__upper_left = shape.upper_left+self.displace
        self.__upper_right = shape.upper_right+self.displace
        self.__lower_left = shape.lower_left+self.displace
        self.__lower_right = shape.lower_right+self.displace

        depthmag = (self.__depth).mag
        if self.__depth.z <= 0: # adjust for case of backward extrusion
            p = [self.displace+vector(0,0,0), self.displace+vector(0,0,-depthmag)]
        else:
            p = [self.displace+vector(0,0,depthmag), self.displace+vector(0,0,0)]
        self.extrusion = extrusion(frame=self.__frame, pos=p, shape=shape.Polygon, 
                  color=self.__color, material=self.__material, twosided=self.__twosided)

    def rotate(self, angle=0, axis=None, origin=None):
        if axis is None:
            axis = self.__display.up
        if origin is None:
            origin = self.__frame.pos
        self.__frame.rotate(origin=vector(origin), angle=angle, axis=vector(axis))
                
    def set_text(self, text):
        if self.__text == text: return
        self.destroyText()
        self.__text = text
        self.paintText()
    def get_text(self):
        return self.__text
                
    def check_align(self, align):
        if align == 'left' or align == 'right' or align == 'center':
            self.__align = align
        else:
            raise ValueError("align must be 'left', 'right', or 'center'") 
                
    def set_align(self, align):
        oldalign = self.__align
        self.check_align(align)
        if align == 'left':
            displace = 0 # nothing needed if current align is "left"
        elif align == 'right':
            displace = -self.__width
        else:
            displace = -self.__width/2
        if oldalign == 'left':
            if align == 'left': return
            displace += 0
        elif oldalign == 'right':
            if align == 'right': return
            displace += self.__width
        else:
            if align == 'center': return
            displace += self.__width/2
        if displace == 0: return
        self.extrusion.pos[...,0] += displace
        
        displace = vector(displace,0,0)
        self.__start += displace
        self.__upper_left += displace
        self.__upper_right += displace
        self.__lower_left += displace
        self.__lower_right += displace
        for s in self.__starts:
            s += displace
    def get_align(self):
        return self.__align
                
    def set_font(self, font):
        if self.__font == font: return
        self.__font = font
        self.paintText()
    def get_font(self):
        return self.__font

    def set_height(self, height):
        height = float(height)
        if height == self.height: return
        oldheight = self.__height
        self.__height = height
        factor = height/oldheight
        self.extrusion.scale *= factor
        self.__descent *= factor
        self.__vertical_spacing *= factor
        self.displace *= factor
        self.extrusion.pos *= factor
        self.__width *= factor
        for n in range(len(self.__widths)):
            self.__widths[n] *= factor
        self.__start *= factor
        self.__upper_left *= factor
        self.__upper_right *= factor
        self.__lower_left *= factor
        self.__lower_right *= factor
        for s in self.__starts:
            s *= factor
    def get_height(self):
        return self.__height

    def set_depth(self, depth):
        olddepth = self.__depth
        try:
            if len(depth) == 3:
                self.__depth = vector(depth)
        except:
            self.__depth = vector(0,0,depth)
        if (self.__depth-olddepth).mag == 0: return
        depthmag = (self.__depth).mag
        if self.__depth.z <= 0: # adjust for case of backward extrusion
            self.extrusion.pos = [self.displace+vector(0,0,0), self.displace+vector(0,0,-depthmag)]
            self.__frame.axis = (vector(0,1,0).cross(-self.__depth)).norm()
        else:
            self.extrusion.pos = [self.displace+vector(0,0,depthmag), self.displace+vector(0,0,0)]
            self.__frame.axis = (vector(0,1,0).cross(self.__depth)).norm()
    def get_depth(self):
        return self.__depth
                
    def set_width(self, temp):
        raise AttributeError("Cannot set the width")
    def get_width(self):
        return self.__width
                
    def set_widths(self, temp):
        raise AttributeError("Cannot set the widths")
    def get_widths(self):
        return self.__widths
                
    def set_descent(self, temp):
        raise AttributeError("Cannot set the descent")
    def get_descent(self):
        return abs(self.__descent)

    def set_spacing(self, spacing):
        if self.__spacing == spacing: return
        self.__spacing = spacing
        self.paintText()
    def get_spacing(self):
        return self.__spacing
                
    def set_vertical_spacing(self, vspace):
        old = self.__vertical_spacing
        if vspace == old: return
        self.__vertical_spacing = vspace
        self.paintText()
    def get_vertical_spacing(self):
        return self.__vertical_spacing
                
    def set_upper_left(self, temp):
        raise AttributeError("Cannot set upper_left")
    def get_upper_left(self):
        return self.__frame.frame_to_world(self.__upper_left)
                
    def set_upper_right(self, temp):
        raise AttributeError("Cannot set upper_right")
    def get_upper_right(self):
        return self.__frame.frame_to_world(self.__upper_right)
                
    def set_lower_left(self, temp):
        raise AttributeError("Cannot set lower_left")
    def get_lower_left(self):
        return self.__frame.frame_to_world(self.__lower_left)
                
    def set_lower_right(self, temp):
        raise AttributeError("Cannot set lower_right")
    def get_lower_right(self):
        return self.__frame.frame_to_world(self.__lower_right)

    def set_start(self, temp):
        raise AttributeError("Cannot set start")
    def get_start(self):
        return self.__frame.frame_to_world(self.__start)

    def set_starts(self, temp):
        raise AttributeError("Cannot set starts")
    def get_starts(self):
        s = []
        for L in self.__starts:
            s.append(self.__frame.frame_to_world(L))
        return s
                
    def set_pos(self, pos):
        self.__frame.pos = vector(pos)
    def get_pos(self):
        return self.__frame.pos

    def set_x(self, x):
        self.__frame.pos.x = x
    def get_x(self):
        return self.__frame.pos.x

    def set_y(self, y):
        self.__frame.pos.y = y
    def get_y(self):
        return self.__frame.pos.y

    def set_z(self, z):
        self.__frame.pos.z = z
    def get_z(self):
        return self.__frame.pos.z

    def set_material(self, material):
        self.extrusion.material = material
    def get_material(self):
        return self.__material

    def set_color(self, color):
        self.extrusion.color = color
        self.__color = color
    def get_color(self):
        return self.__color

    def set_red(self, red):
        self.__color = (red, self.__color[1], self.__color[2])
        self.set_color(self.__color)
    def get_red(self):
        return self.__color[0]

    def set_green(self, green):
        self.__color = (self.__color[0], green, self.__color[2])
        self.set_color(self.__color)
    def get_green(self):
        return self.__color[1]

    def set_blue(self, blue):
        self.__color = (self.__color[0], self.__color[1], blue)
        self.set_color(self.__color)
    def get_blue(self):
        return self.__color[2]

    def set_axis(self, axis):
        self.__frame.axis = vector(axis).norm()
    def get_axis(self):
        return self.__frame.axis.norm()

    def set_display(self, disp):
        self.__display = self.__frame.display = disp
    def get_display(self):
        return self.__display

    def set_frame(self, fr):
        self.__frame = fr
    def get_frame(self):
        return self.__frame

    def set_up(self, up):
        self.__frame.up = up
    def get_up(self):
        return self.__frame.up

    def set_twosided(self, twosided):
        self.__twosided = twosided
    def get_twosided(self):
        return self.__twosided

    def set_visible(self, visible):
        self.__frame.visible = self.__visible = visible
    def get_visible(self):
        return self.__visible

    pos = property( get_pos, set_pos, None)
    x = property( get_x, set_x, None)
    y = property( get_y, set_y, None)
    z = property( get_z, set_z, None)
    text = property( get_text, set_text, None)
    align = property( get_align, set_align, None)
    font = property( get_font, set_font, None)
    height = property( get_height, set_height, None)
    color = property( get_color, set_color, None)
    material = property( get_material, set_material, None)
    red = property( get_red, set_red, None)
    green = property( get_green, set_green, None)
    blue = property( get_blue, set_blue, None)
    axis = property( get_axis, set_axis, None)
    depth = property( get_depth, set_depth, None)
    display = property( get_display, set_display, None)
    frame = property( get_frame, set_frame, None)
    up = property( get_up, set_up, None)
    visible = property( get_visible, set_visible, None)
    width = property( get_width, set_width, None)
    widths = property( get_widths, set_widths, None)
    descent = property( get_descent, set_descent, None)
    vertical_spacing = property( get_vertical_spacing, set_vertical_spacing, None)
    upper_left = property( get_upper_left, set_upper_left, None)
    upper_right = property( get_upper_right, set_upper_right, None)
    lower_left = property( get_lower_left, set_lower_left, None)
    lower_right = property( get_lower_right, set_lower_right, None)
    start = property( get_start, set_start, None)
    starts = property( get_starts, set_starts, None)
    spacing = property( get_spacing, set_spacing, None)
    twosided = property( get_twosided, set_twosided, None)

    
