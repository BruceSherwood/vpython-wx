from math import pi, cos, sin, sqrt, acos
from .cvisual import vector, norm, rotate
from . import shapes as sh

def convert(pos=(0,0,0), up=(0,1,0), points=None, closed=True):
        pos = vector(pos)
        up = norm(vector(up))
        up0 = vector(0,1,0)
        angle = acos(up.dot(up0))
        reorient = (angle > 0.0)
        axis = up0.cross(up)
        pts = []
        for pt in points:
            newpt = vector(pt[0],0,-pt[1])
            if reorient: newpt = newpt.rotate(angle=angle, axis=axis)
            pts.append(pos+newpt)
        if closed and (pts[-1] != pts[0]): pts.append(pts[0])
        return pts

class rectangle():
    def __init__(self, pos=(0,0,0), width=6, height=None, rotate=0.0, thickness=None,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if height == None: height = width
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a rectangular path")
        c = sh.rectangle(width=width, height=height, rotate=rotate, thickness=0,
                  roundness=roundness, invert=invert, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up

class cross():
    def __init__(self, pos=(0,0,0), width=5, thickness=2, rotate=0.0,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        c = sh.cross(width=width, rotate=rotate, thickness=thickness,
                  roundness=roundness, invert=invert, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up

class trapezoid():
    def __init__(self, pos=(0,0,0), width=6, height=3, top=None, rotate=0.0, thickness=None,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if height == None: height = width
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a trapezoidal path")
        c = sh.trapezoid(width=width, height=height, top=top, rotate=rotate, thickness=0,
                  roundness=roundness, invert=invert, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up

class circle():
    def __init__(self, pos=(0,0,0), radius=3, np=32, thickness=None,
                  scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a circular path")
        c = sh.circle(radius=radius, np=np, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up

class line():
    def __init__(self, start=(0,0,0), end=(0,0,-1), np=2):
        if np < 2:
            raise AttributeError("The minimum value of np is 2 (one segment)")
        start = vector(start)
        end = vector(end)
        vline = (end-start)/(np-1)
        self.pos = []
        for i in range(np):
            self.pos.append(start + i*vline)
        self.up = (0,1,0)

class arc():
    def __init__(self, pos=(0,0,0), radius=3, np=32, rotate=0.0, angle1=0, angle2=pi, thickness=None,
                  scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a circular path")
        c = sh.arc(radius=radius, angle1=angle1, angle2=angle2, rotate=rotate, np=np,
                   scale=scale, xscale=xscale, yscale=yscale, path=True)
        self.pos = convert(pos=pos, up=up, points=c[0], closed=False)
        self.up = up

class ellipse():
    def __init__(self, pos=(0,0,0), width=6, height=None, np=32, thickness=None,
                  scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in an elliptical path")
        c = sh.ellipse(width=width, height=height, np=np, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up

class ngon():
    def __init__(self, pos=(0,0,0), np=3, length=6, radius=3.0, rotate=0.0, thickness=None,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in an ngon path")
        c = sh.ngon(np=np, length=length, radius=radius, rotate=rotate, thickness=0,
                  roundness=roundness, invert=invert, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up

class triangle():
    def __init__(self, pos=(0,0,0), np=3, length=6, rotate=0.0, thickness=None,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a triangular path")
        c = sh.ngon(np=np, length=length, rotate=rotate-pi/6.0, thickness=0,
                  roundness=roundness, invert=invert, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up


class pentagon():
    def __init__(self, pos=(0,0,0), np=5, length=6, rotate=0.0, thickness=None,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a pentagonal path")
        c = sh.ngon(np=np, length=length, rotate=rotate+pi/10, thickness=0,
                  roundness=roundness, invert=invert, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up

class hexagon():
    def __init__(self, pos=(0,0,0), np=6, length=6, rotate=0.0, thickness=None,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a hexagonal path")
        c = sh.ngon(np=np, length=length, rotate=rotate, thickness=0,
                  roundness=roundness, invert=invert, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up

class star():
    def __init__(self, pos=(0,0,0), radius=3, n=5, iradius=None, rotate=0.0, thickness=None,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a star path")
        c = sh.star(n=n, radius=radius, iradius=iradius, rotate=rotate,
                  roundness=roundness, invert=invert, scale=scale, xscale=xscale, yscale=yscale)
        self.pos = convert(pos=pos, up=up, points=c.contour(0))
        self.up = up
        
class pointlist():
    def __init__(self, pos=[], rotate=0.0, thickness=None,
                  roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0, up=(0,1,0)):
        if thickness is not None:
            raise AttributeError("Thickness is not allowed in a pointlist path")
        # pos may be either a list of points or a Polygon object
        try:
            points = pos.contour(0)
            if len(pos) > 1:
                raise AttributeError("pointlist can deal with only a single contour.")
        except:
            points = pos[:]
        closed = (points[-1] == points[0])
        if not closed:
            points.append(points[0])
        c = sh.pointlist(pos=points, rotate=rotate, roundness=roundness, invert=invert,
                         scale=scale, xscale=xscale, yscale=yscale, path=True)
        self.pos = convert(pos=(0,0,0), up=up, points=c[0], closed=closed)
        self.up = up
        
