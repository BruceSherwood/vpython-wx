"""
Simulation of the VPython camera geometry.
Version using wx widgets.  Geoff Tovey, England, 11 March 2014.

================================================================================"""

from __future__ import division, print_function
import visual as vs   # for 3D panel 
import wx   # for widgets

# Draw window & 3D pane =================================================

win = vs.window(width=1024, height=720, menus=False, title='SIMULATE VPYTHON GUI')
                         # make a main window. Also sets w.panel to addr of wx window object. 
scene = vs.display( window=win, width=830, height=690, forward=-vs.vector(1,1,2))
                         # make a 3D panel 
clr = vs.color
vss = scene

# Draw 3D model ======================

def axes( frame, colour, sz, posn ): # Make axes visible (of world or frame).
                                     # Use None for world.   
    directions = [vs.vector(sz,0,0), vs.vector(0,sz,0), vs.vector(0,0,sz)]
    texts = ["X","Y","Z"]
    posn = vs.vector(posn)
    for i in range (3): # EACH DIRECTION
       vs.curve( frame = frame, color = colour, pos= [ posn, posn+directions[i]])
       vs.label( frame = frame,color = colour,  text = texts[i], pos = posn+ directions[i],
                                                                    opacity = 0, box = False )

axes( None, clr.white, 3, (-11,6,0))


def drawGrid( posn=(0,0,0), sq=1, H=5, W = 8, normal='z', colour= clr.white ) :
    """ Draw grid of squares in XY, XZ or YZ plane with corner nearest origin at given posn.
    sq= length of side of square.  H = number of squares high (Y). W = number of squares wide (X).
    normal is the axis which is normal to the grid plane. 
    """
    ht = H*sq
    wd = W*sq
    for i in range( 0, wd + 1, sq ):  # FOR EACH VERTICAL LINE
        if   normal == 'z':   vs.curve( pos=[(posn[0]+i, posn[1]+ht, posn[2]),
                                             (posn[0]+i, posn[1],    posn[2])], color=colour )
        elif normal == 'x':   vs.curve( pos=[(posn[0], posn[1]+ht,   posn[2]+i),
                                             (posn[0], posn[1],      posn[2]+i)], color=colour)
        else:                 vs.curve( pos=[(posn[0]+i, posn[1], posn[2]+ht),
                                             (posn[0]+i, posn[1], posn[2])], color=colour)
    for i in range( 0, ht+1, sq ):  # FOR EACH HORIZONTAL LINE
        if normal == 'z':   vs.curve( pos=[(posn[0],    posn[1]+i, posn[2]),
                                           (posn[0]+wd, posn[1]+i, posn[2])], color=colour)
        elif normal == 'x': vs.curve( pos=[(posn[0], posn[1]+i, posn[2]+wd),
                                           (posn[0], posn[1]+i, posn[2])], color=colour)
        else:               vs.curve( pos=[(posn[0],    posn[1], posn[2]+i),
                                           (posn[0]+wd, posn[1], posn[2]+i)], color=colour)

drawGrid( normal = 'z', posn= (-6, 0, -6), colour = clr.blue,   W = 12 )
drawGrid( normal = 'z', posn= (-6, 0,  6), colour = clr.blue,   W = 5 )
drawGrid( normal = 'z', posn= ( 1, 0,  6), colour = clr.blue,   W = 5 )
drawGrid( normal = 'x', posn= (-6, 0, -6), colour = clr.green,  W = 12 )
drawGrid( normal = 'x', posn= ( 6, 0, -6), colour = clr.green,  W = 12 )
drawGrid( normal = 'y', posn= (-6, 0, -6), colour = clr.orange, W = 12, H = 12 )
drawGrid( normal = 'z', posn= (-6, 0,  0), colour = clr.red,    W = 12 )

# The central post. Base of post is the origin. *************** 
pole= vs.cylinder( pos=(0,0,0),axis=(0,3,0), radius=0.1, color=(1,0,0))

scene_size = 12 # approx size of model drawn above.

def drawLine( posn, length, direction):  # draw straight line STARTING at given posn, with given
                                        # length and direction.  ALL ARE RELATIVE TO CAMERA FRAME
    return vs.curve(    frame=cam_frame, pos = [posn, posn+direction.norm()*length ])

def reDrawLine( line, posn, length, direction):
    line.pos = [posn, posn+direction.norm()*length ]


def drawCameraFrame():  # create frame and draw its contents
    global  cam_box, cent_plane,  cam_lab, cam_tri, range_lab, linelen, fwd_line
    global fwd_arrow, mouse_line, mouse_arrow, mouse_lab, fov, range_x, cam_dist, cam_frame
    global ray
    cam_frame = vs.frame( pos = vs.vector(0,2,2,),  axis = (0,0,1))
               # NB: contents are rel to this frame.  start with camera looking "forward"
               # origin is at simulated scene.center
    fov = vs.pi/3.0  # 60 deg 
    range_x = 6  # simulates scene.range.x  
    cam_dist = range_x / vs.tan(fov/2.0)  # distance between camera and center. 
    ray = vs.vector(-20.0, 2.5, 3.0).norm()  # (unit) direction of ray vector (arbitrary)
                                         #  REL TO CAMERA FRAME
    cam_box = vs.box(frame=cam_frame, length=1.5, height=1, width=1.0, color=clr.blue,
                                                   pos=(cam_dist,0,0)) # camera-box
    cent_plane = vs.box(frame=cam_frame, length=0.01, height=range_x*1.3, width=range_x*2,
                                                    pos=(0,0,0), opacity=0.5 )  # central plane
    cam_lab = vs.label(frame=cam_frame, text= 'U', pos= (cam_dist,0,0), height= 9, xoffset= 6)
    cam_tri = vs.faces( frame=cam_frame, pos=[(0,0,0), (0,0,-range_x), (cam_dist,0,0)])
    cam_tri.make_normals()
    cam_tri.make_twosided()
    range_lab = vs.label(frame=cam_frame, text= 'R', pos= (0, 0, -range_x), height= 9, xoffset= 6)
    linelen = scene_size + vs.mag( cam_frame.axis.norm()*cam_dist + cam_frame.pos)
                                                                   # len of lines from camera
    fwd_line = drawLine( vs.vector(cam_dist,0,0), linelen, vs.vector(-1,0,0))
    fwd_arrow = vs.arrow(frame=cam_frame, axis=(-2,0,0), pos=(cam_dist, 0, 0), shaftwidth=0.08,
                                                                            color=clr.yellow)
    vs.label(frame=cam_frame, text='C', pos=(0,0,0), height=9, xoffset=6, color=clr.yellow)
    mouse_line = drawLine ( vs.vector(cam_dist,0,0), linelen, ray ) 
    mouse_arrow = vs.arrow(frame=cam_frame, axis=ray*2, pos=(cam_dist,0,0), shaftwidth=0.08,
                                                                                   color=clr.red)
    mouse_lab = vs.label(frame=cam_frame, text= 'M', height= 9, xoffset= 10, color=clr.red, 
                                pos=  -ray*(cam_dist/vs.dot(ray,(1,0,0))) + (cam_dist,0,0))

drawCameraFrame()
# axes( cam_frame, clr.red, 3, (11, 6, 0)) # testing ###########

############## cam_frame.pos       simulates scene.center     ################
############## - cam_frame.axis    simulates scene.forward    ################
############## range_x             simulates scene.range.x    ################
############## fov                 simulates scene.fov        ################
#### ray converted to world coords simulates scene.mouse.ray  ################

#  Animation tools ==========================================

def reDrawLines():
   global fwd_line, mouse_line, cam_dist, ray, scene_size, linelen
   linelen = scene_size + vs.mag( cam_frame.axis.norm()*cam_dist + cam_frame.pos)
   reDrawLine( fwd_line, vs.vector(cam_dist,0,0), linelen, vs.vector(-1,0,0))
   reDrawLine( mouse_line, vs.vector(cam_dist,0,0), linelen, ray ) 

def reDrawTri():  # redraw the camera triangle
   global cam_tri, range_x, cam_dist
   cam_tri.pos = [(0,0,0), (0,0,-range_x), (cam_dist,0,0)]
   cam_tri.make_normals()
   cam_tri.make_twosided()


# Event handlers =========================

def setModView():  # set so that we see view from mod-cam
    global saved_pyvars
    vss.userspin = vss.userzoom = False
    vss.autoscale = vss.autocenter = False  # should not be necessary,  but is! 
    saved_pyvars = [ tuple(vss.forward), tuple(vss.center), vss.fov ]
    # save VPython GUI status (so that we can restore it later ). tuple()is NEEDED so the data
    # is copied - not just its address.   vss.range is not useful. 
    vss.forward = - cam_frame.axis 
    vss.center =  cam_frame.pos
    vss.fov = fov
    vss.range = range_x
    cam_box.visible = fwd_arrow.visible = mouse_arrow.visible = False
    cam_tri.visible = False    
    
def setPyView():  # set so we see view from py-cam (ie std VPython)  
    vss.userspin = vss.userzoom = True
    vss.forward, vss.center, vss.fov = saved_pyvars
                                    # Restore py-vars to what they were when qPy was turned off.
                                    # Except RANGE - as cannot be saved.  So.... 
    vss.range = scene_size*1.5   # SET it.  
    cam_box.visible = fwd_arrow.visible = mouse_arrow.visible = True
    cam_tri.visible = True   


def hCamera(evt): # re "Switch Camera" button
    global qPy
    if qPy:  # we are seeing view from py-cam 
       qPy = False     
       setModView()  # set so that we see view from mod-cam
    else:          
       qPy = True
       setPyView()  # set so we see view from py-cam (ie std VPython)
               
def hReset(evt): # re "Reset" button
    global cam_frame
    cam_box.visible = fwd_arrow.visible = mouse_arrow.visible = True
    cam_tri.visible = True  # so is included in cam_frame.objects list.
    for obj in cam_frame.objects:
       obj.visible = False
       del obj
    del cam_frame
    drawCameraFrame()  # recreate camera frame and its contents
    mode_lab.SetLabel("")  # as is no longer right 
    if not qPy:  setModView() # because drawCameraFrame() assumes qPy is True. 

def hRadio(evt): # re radio button
    global mode
    mode =  ['c', 'f', 'r', 'v', 'm', None][ bRadio.GetSelection()]
    if mode == None:  mode_lab.SetLabel('') 

def hHelp(evt): # re "HELP" button
    wx.MessageBox(
"""The BLUE BOX (at U) represents the camera - ie your viewpoint in the model.
The WHITE rectangle represents the VPython window on your display device.
      On that: C marks its centre (held as scene.center)
                    R marks the midpoint of the right-hand edge, and
                    M marks the mouse position (held as scene.mouse.pos, READ ONLY).
The YELLOW ARROW marks the scene.forward (unit) vector.
The RED ARROW marks the scene.mouse.ray (unit) vector.  It is READ ONLY.
The right-angled triangle U-C-R shows the relationship between the main scene attributes:  
      The shape is determined by the angle at U - which is held as scene.fov/2.
      The size is determined by the length C-R - which is held as scene.range.x
      The position is determined by the position of C - which is held as scene.center.
      The orientation is determined by the direction U-C - which is held as scene.forward.
   
The camera position is also held (as scene.mouse.camera) - but is READ ONLY.
In this simulation scene.range is the same on all 3 axes.

The control panel lets you see the effect of altering each of the above scene attributes. 
The "Switch camera" button shows you the view from the camera in the model. It toggles.
The "Reset model" button resets the model to its initial state.  It does not alter your view.  

Zooming is similar to altering scene.range but does not alter scene.range.  
Spinning is similar to altering scene.forward.
""", 'HELP',  wx.OK  )   

def hMousedown():  # handle mouse-DOWN event
    global qDragging, mouse_pos_old
    vss.unbind( 'mousedown', hMousedown) # STOP monitoring for mouse-down
    qDragging = True
    mouse_pos_old = vss.mouse.pos  # forget old position of mouse. 
    vss.bind( 'mouseup', hMouseup)  # START monitoring for mouse-up   

vss.bind( 'mousedown', hMousedown) # START monitoring for mouse-down 

def hMouseup():  # handle mouse-UP event
    global qDragging
    vss.unbind( 'mouseup', hMouseup)  # STOP monitoring for mouse-up
    qDragging = False
    vss.bind( 'mousedown', hMousedown) # START monitoring for mouse-down 
  
# Draw widgets ==========================
    
x1 = scene.width + 5 
pan = win.panel   # addr of wx window object 
pan.SetSize( (1024,720)) # work-around to make display work.  >= size of client area of window.  

# Controls (= widgets) have to be put in the wx window. 
# Positions and sizes are in pixels, and pos(0,0) is the upper left corner of the window.

wx.StaticText( pan, pos=(x1,10),
    label = "Select an item, then\nleft-drag the mouse over\nthe model "
            "to see the\neffect of altering that:" )

bRadio = wx.RadioBox(pan, pos=(x1,90), choices = ['scene.center', 'scene.forward', 'scene.range',
          'scene.fov', 'moving the mouse', 'NONE OF THOSE'], style= wx.RA_VERTICAL)

bRadio.SetSelection(5)   # Set NONE as intially selected.                
             
bRadio.Bind(wx.EVT_RADIOBOX, hRadio)
mode_lab = wx.StaticText( pan, pos=(x1,310))  

def str2(object):  # convert tuple or vector to char values with 2 sig figs.  Ie is like str()
                   # but shows only 2 sig figs.  
   RC = '' 
   for i in object:
      RC += format( i, " 9.2")  # Means output has: pad LHS with spaces, width 9, 2 sig figs. 
   return RC  

def Button1( label, y, func):
   bb = wx.Button( pan, label=label, pos=(x1+5,y), size = (150,40))
   bb.Bind(wx.EVT_BUTTON, func)

Button1('HELP',          500, hHelp ) 
Button1('Switch Camera', 550, hCamera )
Button1('Reset model',   600, hReset )

# Capture events  =========================================

mode = None
qDragging = False 
qPy = True # we see view from std VPython camera
mouse_pos_old = vss.mouse.pos
vss.autoscale = False
vss.autocenter = False

while True:
    vs.rate(20)
    if not qDragging:  continue  # only simulate changes if left button is down
    if vss.mouse.pos == mouse_pos_old: continue
    mouse_change = (vss.mouse.pos - mouse_pos_old)
    mouse_pos_old = vss.mouse.pos
  
    if mode == 'c':  # demonstrate altering scene.center
       mouse_change.z = mouse_change.x  # improve variation 
       cam_frame.pos = cam_frame.pos + mouse_change/1.5
       reDrawLines()
       mode_lab.SetLabel('scene.center:\n' + str2(cam_frame.pos))
       if not qPy:  vss.center = cam_frame.pos               

    elif mode == 'f': # demonstrate altering scene.forward vector     
       cam_frame.axis = (cam_frame.axis + mouse_change/12.0).norm()  
       cam_frame.up = (0,1,0)
       mode_lab.SetLabel('scene.forward:\n' + str2( - cam_frame.axis))
       if not qPy:  vss.forward = - cam_frame.axis                
       
    elif mode == 'r': # demonstrate altering scene.range.  Alters size of camera triangle.
        if qPy: gearing = 4
        else: gearing = 1 
        cam_dist = cam_dist + (mouse_change.x + mouse_change.y + mouse_change.z)*gearing
        if cam_dist <= 0:  cam_dist = 0.001  # allow only positive
        limit = scene_size*2
        if cam_dist > limit: cam_dist = limit # limit size 
        reDrawLines()
        cam_box.pos = (cam_dist,0,0) 
        cam_lab.pos = (cam_dist,0,0) 
        fwd_arrow.pos = (cam_dist,0,0)
        mouse_arrow.pos = (cam_dist,0,0)
        mouse_lab.pos = -ray*(cam_dist/vs.dot(ray,(1,0,0))) + (cam_dist,0,0)
        range_x = cam_dist * vs.tan(fov/2.0)
        cent_plane.width = range_x*2
        cent_plane.height = range_x*4.0/3
        reDrawTri()  # redraw the camera triangle
        range_lab.pos= (0,0,-range_x)
        mode_lab.SetLabel( 'scene.range:\n' + format( range_x, "9.3")) 
        if not qPy:  vss.range = range_x

    elif mode == 'v': # demonstrate altering scene.fov
        cam_dist = cam_dist + (mouse_change.x + mouse_change.y + mouse_change.z)*4
        if cam_dist <= 0:  cam_dist = 0.001  # allow only positive
        ray = (mouse_lab.pos - (cam_dist, 0,0)).norm()  
        reDrawLines()
        cam_box.pos = (cam_dist,0,0) 
        cam_lab.pos = (cam_dist,0,0) 
        fwd_arrow.pos = (cam_dist,0,0)
        mouse_arrow.pos = (cam_dist,0,0)
        mouse_arrow.axis = ray*2
        reDrawTri()  # redraw the camera triangle
        fov = 2*vs.arctan( range_x / cam_dist)
        mode_lab.SetLabel( 'scene.fov:\n{0:9.0f} deg'.format( vs.degrees(fov))) 
        if not qPy: vss.fov = fov  

    elif mode == 'm': # demonstrate moving the mouse.
        hit = vss.mouse.project( cam_frame.axis, cam_frame.pos)
        m_pos = cam_frame.world_to_frame(hit)
        if abs(m_pos.z) <= cent_plane.width/2.0 and \
           abs(m_pos.y) <= cent_plane.height/2.0 :   # not "off the screen" 
            ray = (m_pos - (cam_dist,0,0)).norm()
            reDrawLine( mouse_line, vs.vector(cam_dist,0,0), linelen, ray)
            mouse_lab.pos = m_pos
            mouse_arrow.axis = 2*ray
            rayout = cam_frame.frame_to_world(ray) - cam_frame.frame_to_world((0,0,0)) 
            mode_lab.SetLabel( 'scene.mouse.ray:\n' + str2( rayout))
            
            


