from __future__ import division
from math import pi, cos, sin, sqrt, tan, atan, acos
try:
    from Polygon import Polygon
    from Polygon.Shapes import Star
    from ttfquery import describe, glyphquery, glyph
except:
    pass

from .cvisual import vector, dot, mag, norm
from .cvisual import cross as vector_cross

import os, sys, glob

Default = "default"

## The following code, from here to the end of findSystemFonts, is from
## https://mail.enthought.com/pipermail/enthought-svn/2005-September/000724.html
## Authors: John Hunter <jdhunter at ace.bsd.uchicago.edu>
##          Paul Barrett <Barrett at STScI.Edu>

#  OS Font paths
MSFolders = \
    r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'

MSFontDirectories   = [
    r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts',
    r'SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts']

X11FontDirectories  = [
    # what seems to be the standard installation point
    "/usr/X11R6/lib/X11/fonts/TTF/",
    # documented as a good place to install new fonts...
    "/usr/share/fonts/",
    # common application, not really useful
    "/usr/lib/openoffice/share/fonts/truetype/",
    ]

OSXFontDirectories = [
    "/Library/Fonts/",
    "/Network/Library/Fonts/",
    "/System/Library/Fonts/"
]

home = os.environ.get('HOME')
if home is not None:
    # user fonts on OSX
    path = os.path.join(home, 'Library', 'Fonts')
    OSXFontDirectories.append(path)

def win32FontDirectory():
    """Return the user-specified font directory for Win32."""

    imported = False
    try:
        import winreg as wreg
        imported = True
    except ImportError:
        pass
    if not imported:
        try:
            import _winreg as wreg
            imported = True
        except ImportError:
            return os.path.join(os.environ['WINDIR'], 'Fonts')
    if imported:
        user = wreg.OpenKey(wreg.HKEY_CURRENT_USER, MSFolders)
        try:
            return wreg.QueryValueEx(user, 'Fonts')[0]
        finally:
            wreg.CloseKey(user)
    return None

def win32InstalledFonts(directory=None, fontext='ttf'):
    """Search for fonts in the specified font directory, or use the
system directories if none given.  A list of TrueType fonts are
returned by default with AFM fonts as an option.
"""

    if directory is None:
        directory = win32FontDirectory()

    imported = False
    try:
        import winreg as wreg
        imported = True
    except ImportError:
        pass
    if not imported:
        try:
            import _winreg as wreg
            imported = True
        except ImportError:
            raise ImportError("Cannot find winreg module")

    key, items = None, {}
    for fontdir in MSFontDirectories:
        try:
            local = wreg.OpenKey(wreg.HKEY_LOCAL_MACHINE, fontdir)
        except OSError:
            continue

        if not local:
            return glob.glob(os.path.join(directory, '*.'+fontext))
        try:
            for j in range(wreg.QueryInfoKey(local)[1]):
                try:
                    key, direc, any = wreg.EnumValue( local, j)
                    if not os.path.dirname(direc):
                        direc = os.path.join(directory, direc)
                    direc = os.path.abspath(direc).lower()
                    if direc[-4:] == '.'+fontext:
                        items[direc] = 1
                except EnvironmentError:
                    continue
                except WindowsError:
                    continue

            return list(items.keys())
        finally:
            wreg.CloseKey(local)
    return None

def OSXFontDirectory():
    """Return the system font directories for OS X."""

    fontpaths = []
    for fontdir in OSXFontDirectories:
        try:
            if os.path.isdir(fontdir):
                for root, dirs, files in os.walk(fontdir):
                    fontpaths.append(root)
        except (IOError, OSError, TypeError, ValueError):
            pass
    return fontpaths

def OSXInstalledFonts(directory=None, fontext=None):
    """Get list of font files on OS X - ignores font suffix by default"""
    if directory is None:
        directory = OSXFontDirectory()

    files = []
    for path in directory:
        if fontext is None:
            files.extend(glob.glob(os.path.join(path,'*')))
        else:
            files.extend(glob.glob(os.path.join(path, '*.'+fontext)))
            files.extend(glob.glob(os.path.join(path, '*.'+fontext.upper())))
    return files


def x11FontDirectory():
    """Return the system font directories for X11."""
    fontpaths = []
    for fontdir in X11FontDirectories:
        try:
            if os.path.isdir(fontdir):
                for root, dirs, files in os.walk(fontdir):
                    fontpaths.append(root)
        except (IOError, OSError, TypeError, ValueError):
            pass
    return fontpaths

def findSystemFonts(fontpaths=None, fontext='ttf'):

    """Search for fonts in the specified font paths, or use the system
paths if none given.  A list of TrueType fonts are returned by default
with AFM fonts as an option.
"""

    fontfiles = {}

    if fontpaths is None:

        if sys.platform == 'win32':
            fontdir = win32FontDirectory()

            fontpaths = [fontdir]
            # now get all installed fonts directly...
            for f in win32InstalledFonts(fontdir):
                base, ext = os.path.splitext(f)
                if len(ext)>1 and ext[1:].lower()==fontext:
                    fontfiles[f] = 1
        else:
            fontpaths = x11FontDirectory()
            # check for OS X & load its fonts if present
            if sys.platform == 'darwin':
                for f in OSXInstalledFonts(fontext=fontext):
                    fontfiles[f] = 1

    elif isinstance(fontpaths, str):
        fontpaths = [fontpaths]

    for path in fontpaths:
        files = glob.glob(os.path.join(path, '*.'+fontext))
        files.extend(glob.glob(os.path.join(path, '*.'+fontext.upper())))
        for fname in files:
            fontfiles[os.path.abspath(fname)] = 1

    return [fname for fname in list(fontfiles.keys()) if os.path.exists(fname)]

def findFont(font):
    flist = findSystemFonts()
    if font == "serif":
        font = "Times New Roman"
    elif font == "sans-serif" or font == "sans":
        font = "Verdana"
    elif font == "monospace":
        font = "Courier New"
    f = font.split('.')
    if f[-1] == 'ttf':
        font = font[:-4]
    font = font.split('/')[-1]
    font = font.split('\\')[-1]
    font = font.lower()+".ttf"
    match = None
    sans = None
    freesans = freeserif = freemono = None
    for f in flist:
        if f[-4:] != ".ttf": continue
        if sys.platform == 'win32':
            name = f.split('\\')[-1].lower()
        else:
            name = f.split('/')[-1].lower()
        if name == font:
            match = f
            break
        elif name == "verdana.ttf":
            sans = f
        elif name == "freesans.ttf": # Ubuntu Linux
            freesans = f
        elif name == "freeserif.ttf": # Ubuntu Linux
            freeserif = f
        elif name == "freemono.ttf": # Ubuntu Linux
            freemono = f
        elif len(font) > len(name):
            if name[:-4] == font[:len(name)-4]:
                if match is None:
                    match = f
                elif len(match) < len(name):
                    match = f
        elif len(name) > len(font):
            if font[:-4] == name[:len(font)-4]:
                if match is None:
                    match = f
                elif len(match) < len(font):
                    match = f
    if freesans and not sans:
        sans = freesans
    if freesans and not match and font.lower()[:7] == "verdana":
        match = freesans
    elif freeserif and not match and font.lower()[:5] == "times":
        match = freeserif
    elif freemono and not match and font.lower()[:7] == "courier":
        match = freemono
    if match is not None:
        return match
    elif sans:
        return sans
    else:
        raise ValueError('Cannot find font "'+font+'"')

def rotatep(p, pr, angle):
        '''Rotate a single point p angle radians around pr'''
        sinr, cosr = sin(angle), cos(angle)
        x, y, z = p
        xRel, yRel, zRel = pr
        newx = x * cosr - y * sinr - xRel * cosr + yRel * sinr + xRel
        newy = x * sinr + y * cosr - xRel * sinr - yRel * cosr + yRel
        pr = (newx, newy)
        return pr

def rotatecp(cp, pr, angle):
        '''Rotate point-set cp angle radians around pr'''
        sinr, cosr = sin(angle), cos(angle)
        cpr = []
        for p in cp:
                x, y = p
                xRel, yRel = pr
                newx = x * cosr - y * sinr - xRel * cosr + yRel * sinr + xRel
                newy = x * sinr + y * cosr - xRel * sinr - yRel * cosr + yRel
                cpr.append((newx, newy))
        return cpr

def roundc(cp, roundness=0.1, nseg=8, invert=False):

    vort = 0.0
    cp.pop()
    for i in range(len(cp)):
        i1 = (i+1)%len(cp)
        i2 = (i+2)%len(cp)
        v1 = vector(cp[i1]) - vector(cp[i])
        v2 = vector(cp[(i2)%len(cp)]) - vector(cp[i1])
        dv = dot(v1,v2)
        vort += dv

    if vort > 0: cp.reverse()

    l = 999999
    
    for i in range(len(cp)):
        p1 = vector(cp[i])
        p2 = vector(cp[(i+1)%len(cp)])
        lm = mag(p2-p1)
        if lm < l: l = lm

    r = l*roundness
    ncp = []
    lcp = len(cp)

    for i in range(lcp):
        i1 = (i+1)%lcp
        i2 = (i+2)%lcp
        
        w0 = vector(cp[i])
        w1 = vector(cp[i1])
        w2 = vector(cp[i2])

        wrt = vector_cross((w1-w0),(w2-w0))

        v1 = w1-w0
        v2 = w1-w2
        rax = norm(((norm(v1)+norm(v2))/2.0))
        angle = acos(dot(norm(v2),norm(v1)))
        afl = 1.0
        if wrt[2] > 0: afl = -1.0
        angle2 = angle/2.0
        cc = r/sin(angle2)
        ccp = vector(cp[i1]) - rax*cc
        tt = r/tan(angle2)
        t1 = vector(cp[i1]) -norm(v1)*tt
        t2 = vector(cp[i1]) -norm(v2)*tt

        ncp.append(tuple(t1)[0:2])
        nc = []
        a = 0
        dseg = afl*(pi-angle)/nseg
        if not invert:
            for i in range(nseg):
                nc.append(rotatep(t1, ccp, a))
                ncp.append(tuple(nc[-1])[0:2])
                a -= dseg
        else:
            dseg = afl*(angle)/nseg
            for i in range(nseg):
                nc.append(rotatep(t1, (cp[i1][0],cp[i1][1],0), a))
                ncp.append(tuple(nc[-1])[0:2])
                a += dseg
        ncp.append(tuple(t2)[0:2])
    ncp.append(ncp[0])
    return ncp

def rectangle(pos=(0,0), width=1.0, height=None, rotate=0.0, thickness=0, 
              roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):
        if height is None: height = width
        
        if thickness == 0:
            p0 = (pos[0]+width/2.0, pos[1]-height/2.0)
            p1 = (pos[0]+width/2.0, pos[1]+height/2.0)
            p2 = (pos[0]-width/2.0, pos[1]+height/2.0)
            p3 = (pos[0]-width/2.0, pos[1]-height/2.0)
            p4 = (pos[0]+width/2.0, pos[1]-height/2.0)

            cp = [p0, p1, p2, p3, p4]
            if rotate != 0.0: cp = rotatecp(cp, pos, rotate)
            if scale != 1.0: xscale = yscale = scale
            pp = Polygon(cp)
            if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
            if roundness > 0:
                cp = roundc(pp.contour(0), roundness=roundness, invert=invert)
                return Polygon(cp)
            else: return pp
        else:
            pp = rframe(pos=pos, width=width, height=height, thickness=thickness,
                       rotate=rotate, roundness=roundness, invert=invert,
                       scale=scale, xscale=xscale, yscale=yscale)
            return pp

def cross(pos=(0,0), width=1.0, thickness=0.2, rotate=0.0,
              roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):

        fsqr = rectangle(pos=pos, width=width)
        sqr1 = rectangle(pos=(pos[0]-(width+thickness)/4.0,
                                     pos[1]+(width+thickness)/4.0), width=(width-thickness)/2.0)
        sqr2 = rectangle(pos=(pos[0]+(width+thickness)/4.0,
                                     pos[1]+(width+thickness)/4.0), width=(width-thickness)/2.0)
        sqr3 = rectangle(pos=(pos[0]+(width+thickness)/4.0,
                                     pos[1]-(width+thickness)/4.0), width=(width-thickness)/2.0)
        sqr4 = rectangle(pos=(pos[0]-(width+thickness)/4.0,
                                     pos[1]-(width+thickness)/4.0), width=(width-thickness)/2.0)
        poly = fsqr - sqr1 -sqr2 -sqr3 -sqr4
        cp = poly.contour(0)
        cp.append(cp[0])
        if rotate != 0.0: cp = rotatecp(cp, pos, rotate)
        pp = Polygon(cp)
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        if roundness > 0:
                cp = roundc(pp.contour(0), roundness=roundness, invert=invert)
                return Polygon(cp)
        else: return pp

def rframe(pos=(0,0), width=1.0, height=None, thickness=None, rotate=0.0,
              roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):
        if height == None: height = width
        if thickness == Default or thickness == None: thickness = min(height,width)*0.2
        else: thickness = min(height,width)*thickness*2
        fsqr = rectangle(pos=pos, width=width, height=height)
        sqr1 = rectangle(pos=pos, width=(width-thickness), height=height-thickness)
        pp = fsqr - sqr1
        if rotate != 0.0: pp.rotate(rotate)
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        if roundness > 0:
                cp0 = pp.contour(0)
                cp0.append(cp0[0])
                cp0 = roundc(cp0, roundness=roundness, invert=invert)
                cp1 = pp.contour(1)
                cp1.append(cp1[0])
                cp1 = roundc(cp1, roundness=roundness, invert=invert)
                p1 = Polygon(cp0)
                p2 = Polygon(cp1)
                pp = p2-p1
                return pp
        else: return pp


def nframe(pos=(0,0), length=1.0, np=3, thickness=None, rotate=0.0,
              roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):
        if thickness == Default or thickness == None: thickness = length*0.2
        else: thickness = length*thickness*2
        fsqr = ngon(pos=pos, np=np, length=length)
        nga = (np-2)*pi/np
        angle = (pi/2 - nga)
        length2=length-thickness*cos(angle)
        csa = cos(angle)
        sqr1 = ngon(pos=pos, np=np, length=length-thickness/csa*2-thickness*tan(angle)*2)
        pp = fsqr - sqr1
        if rotate != 0.0: pp.rotate(rotate)
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        if roundness > 0:
                cp0 = pp.contour(0)
                cp0.append(cp0[0])
                cp0 = roundc(cp0, roundness=roundness, invert=invert)
                cp1 = pp.contour(1)
                cp1.append(cp1[0])
                cp1 = roundc(cp1, roundness=roundness, invert=invert)
                p1 = Polygon(cp0)
                p2 = Polygon(cp1)
                pp = p2-p1
                return pp
        else: return pp


def sframe(pos=(0,0), n=5, radius=1.0, iradius=None, thickness=None, rotate=0.0,
              roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):
        if iradius == None: iradius = 0.5*radius
        if thickness == None: thickness = 0.2*radius
        else: thickness = thickness*2*iradius
        fsqr = star(pos=pos, n=n, radius=radius, iradius=iradius)
        angle = pi/2 - (n-2)*pi/n
        sqr1 = star(pos=pos, n=n, radius=radius-thickness, iradius=(radius-thickness)*iradius/radius)
        pp = fsqr - sqr1
        if rotate != 0.0: pp.rotate(rotate)
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        if roundness > 0:
                cp0 = pp.contour(0)
                cp0.append(cp0[0])
                cp0 = roundc(cp0, roundness=roundness, invert=invert)
                cp1 = pp.contour(1)
                cp1.append(cp1[0])
                cp1 = roundc(cp1, roundness=roundness, invert=invert)
                p1 = Polygon(cp0)
                p2 = Polygon(cp1)
                pp = p2-p1
                return pp
        else: return pp

def trapezoid(pos=(0,0), width=2.0, height=1.0, top=None, rotate=0.0, thickness=0,
              roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):
        if top == None: top = width/2.0
        if thickness == 0:
            p0 = (pos[0]+width/2.0, pos[1]-height/2.0)
            p1 = (pos[0]+top/2.0, pos[1]+height/2.0)
            p2 = (pos[0]-top/2.0, pos[1]+height/2.0)
            p3 = (pos[0]-width/2.0, pos[1]-height/2.0)
            p4 = (pos[0]+width/2.0, pos[1]-height/2.0)

            cp = [p0, p1, p2, p3, p4]
            if rotate != 0.0: cp = rotatecp(cp, pos, rotate)
            if scale != 1.0: xscale = yscale = scale
            pp = Polygon(cp)
            if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
            if roundness > 0:
                    cp = roundc(pp.contour(0), roundness=roundness, invert=invert)
                    return Polygon(cp)
            else: return pp
        else:
            pp = trframe(pos=pos, width=width, height=height, thickness=thickness,
                       rotate=rotate, roundness=roundness, invert=invert,
                       top=top, scale=scale, xscale=xscale, yscale=yscale)
            return pp

def trframe(pos=(0,0), width=2.0, height=1.0, top=None, thickness=None, rotate=0.0,
              roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):
        if top == None: top = width/2.0
        if thickness == Default or thickness == None: thickness = min(height,top)*0.2
        else: thickness = min(height,top)*thickness*2
        fsqr = trapezoid(pos=pos, width=width, height=height, top=top)
        angle = atan((width-top)/2.0/height)
        db = (thickness)/cos(angle)
        sqr1 = trapezoid(pos=pos, width=(width-db-thickness*tan(angle)),
                                height=height-thickness, top=top-(db-thickness*tan(angle)))
        pp = fsqr - sqr1
        if rotate != 0.0: pp.rotate(rotate)
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        if roundness > 0:
                cp0 = pp.contour(0)
                cp0.append(cp0[0])
                cp0 = roundc(cp0, roundness=roundness, invert=invert)
                cp1 = pp.contour(1)
                cp1.append(cp1[0])
                cp1 = roundc(cp1, roundness=roundness, invert=invert)
                p1 = Polygon(cp0)
                p2 = Polygon(cp1)
                pp = p2-p1
                return pp
        else: return pp

def circle(pos=(0,0), radius=0.5, np=32, scale=1.0, xscale=1.0, yscale=1.0,
            thickness=0, angle1=0, angle2=2*pi, rotate=0):
        if thickness == 0 or angle1 != 0 or angle2 != 2*pi:
            cp = []
            if angle1 != 0 or angle2 != 2*pi:
                cp.append(pos)
            seg = 2.0*pi/np
            nseg = int(abs((angle2-angle1)/seg+.5))
            seg = (angle2-angle1)/nseg
            if angle1 != 0 or angle2 != 2*pi: nseg += 1
            c = radius*cos(angle1)
            s = radius*sin(angle1)
            dc = cos(seg)
            ds = sin(seg)
            x0 = pos[0]
            y0 = pos[1]
            cp.append((x0+c,y0+s))
            for i in range(nseg-1):
                c2 = c*dc - s*ds
                s2 = s*dc + c*ds
                cp.append((x0+c2,y0+s2))
                c = c2
                s = s2
            if angle1 != 0 or angle2 != 2*pi: cp.append(cp[0])
            if rotate != 0.0 and angle1 != 0 or angle2 != 2*pi:
                cp = rotatecp(cp, pos, rotate)
            if scale != 1.0: xscale = yscale = scale
            pp = Polygon(cp)
            if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
            return pp
        else:
            if thickness == Default: thickness = radius*0.2
            pp = ring(pos=pos, radius=radius, np=np, scale=scale,
                      iradius=(radius-thickness), xscale=xscale, yscale=yscale)
            return pp

def line(pos=(0,0), np=2, rotate=0.0, scale=1.0, xscale=1.0, yscale=1.0,
           thickness=None, start=(0,0), end=(0,1), path=False):
        v = vector((end[0]-start[0]), (end[1]-start[1]))
        if thickness is None:
            thickness = 0.01*mag(v)
        dv = thickness*norm(vector_cross(vector(0,0,1),v))
        dx = dv.x
        dy = dv.y
        cp = [] # outer line
        cpi = [] # inner line
        vline = (vector(end)-vector(start)).norm()
        mline = mag(vector(end)-vector(start))
        for i in range(np):
            x = start[0] + (vline*i)[0]/float(np-1)*mline
            y = start[1] + (vline*i)[1]/float(np-1)*mline
            cp.append( (x+pos[0],y+pos[1]) )
            cpi.append( (x+pos[0]+dx,y+pos[1]+dy) )
        if not path:
                cpi.reverse()
                for p in cpi:
                    cp.append(p)
                cp.append(cp[0])
        if rotate != 0.0: cp = rotatecp(cp, pos, rotate)
        if scale != 1.0: xscale = yscale = scale
        pp = Polygon(cp)
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        if not path:
                return pp
        else:
                return [cp]

def arc(pos=(0,0), radius=0.5, np=32, rotate=0.0, scale=1.0, xscale=1.0, yscale=1.0,
           thickness=None, angle1=0.0, angle2=pi, path=False):
        if thickness is None:
            thickness = 0.01*radius
        cp = []  # outer arc
        cpi = [] # inner arc
        seg = 2.0*pi/np
        nseg = int(abs((angle2-angle1))/seg)+1
        seg = (angle2-angle1)/nseg
        for i in range(nseg+1):
            x = cos(angle1+i*seg)
            y = sin(angle1+i*seg)
            cp.append( (radius*x+pos[0],radius*y+pos[1]) )
            cpi.append( ((radius-thickness)*x+pos[0],(radius-thickness)*y+pos[1]) )
        if not path:
                cpi.reverse()
                for p in cpi:
                    cp.append(p)
                cp.append(cp[0])
        if rotate != 0.0: cp = rotatecp(cp, pos, rotate)
        if scale != 1.0: xscale = yscale = scale
        pp = Polygon(cp)
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        if not path:
                return pp
        else:
                return [cp]

def ring(pos=(0,0), radius=1.0, iradius=None, np=32, scale=1.0, xscale=1.0, yscale=1.0):
        if iradius == None: iradius = radius*0.8
        c1 = circle(pos=pos, radius=radius, np=np, scale=1.0, xscale=1.0, yscale=1.0)
        c2 = circle(pos=pos, radius=iradius, np=np, scale=1.0, xscale=1.0, yscale=1.0)
        pp = c1-c2
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        return pp

def ellipse(pos=(0,0), width=1.0, height=None, np=32, rotate=0.0, thickness=None,
            scale=1.0, xscale=1.0, yscale=1.0, angle1=0, angle2=2*pi):
        if height == None: height = 0.5*width
        if thickness == 0 or thickness == None or angle1 != 0 or angle2 != 2*pi:
            cp = []
            if angle1 != 0 or angle2 != 2*pi:
                cp.append(pos)
            angle = angle1
            radius=0.5
            lf = width/2.0
            hf = height/2.0
            seg = 2.0*pi/np
            nseg = int(abs((angle2-angle1))/seg)+1
            seg = (angle2-angle1)/nseg
            for i in range(nseg+1):
                x = cos(angle)*lf + pos[0]
                y = sin(angle)*hf + pos[1]
                cp.append((x,y))
                angle += seg
            cp.append(cp[0])
            if rotate != 0.0: cp = rotatecp(cp, pos, rotate)
            if scale != 1.0: xscale = yscale = scale
            pp = Polygon(cp)
            if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
            return pp
        else:
            pp = ering(pos=pos, width=width, height=height, np=np, rotate=rotate,
                  thickness=thickness)
            return pp

def ering(pos=(0,0), width=1.0, height=None, np=32, scale=1.0, thickness=None, 
          xscale=1.0, yscale=1.0, rotate=0.0):
        if height == None: height = 0.5*width
        if thickness == Default or thickness == None: thickness = min(width,height)*0.2
        else: 
            t = 0.5*min(width,height)
            if thickness > t:
                thickness = t
        c1 = ellipse(pos=pos, width=width, height=height, np=np,
                     scale=1.0, xscale=1.0, yscale=1.0, rotate=rotate)
        c2 = ellipse(pos=pos, width=width-2*thickness, height=height-2*thickness,
                     np=np, scale=1.0, xscale=1.0, yscale=1.0, rotate=rotate)
        pp = c1-c2
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        return pp
        
def ngon(pos=(0,0), np=3, length=None, radius=1.0, rotate=0.0, thickness=0,
         roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):
        cp = [] 
        if np < 3:
                raise AttributeError("number of sides can not be less than 3")
                return None

        angle = 2*pi/np
        if length != None: radius = (length/2.0)/(sin(angle/2))    
        else: length = radius*(sin(angle/2))*2
        if thickness == 0:
            seg = 2.0*pi/np
            angle = rotate
            for i in range(np):
                x = radius*cos(angle) + pos[0]
                y = radius*sin(angle) + pos[1]
                cp.append((x,y))
                angle += seg
            cp.append(cp[0])
            if scale != 1.0: xscale = yscale = scale
            pp = Polygon(cp)
            if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
            if roundness > 0:
                    cp = roundc(pp.contour(0), roundness=roundness, invert=invert)
                    return Polygon(cp)
            else: return pp
        else:
            pp = nframe(pos=pos, length=length, thickness=thickness, roundness=roundness,
                        invert=invert, rotate=rotate, np=np)
            return pp

def triangle(pos=(0,0), length=1.0, rotate=0.0, roundness=0.0, thickness=0,
             invert=False, scale=1.0, xscale=1.0, yscale=1.0):
    tri = ngon(pos=pos, np=3, length=length, rotate=rotate-pi/6.0,
               roundness=roundness, invert=invert, scale=scale,
               xscale=xscale, yscale=yscale, thickness=thickness)
    return Polygon(tri)

def pentagon(pos=(0,0), length=1.0, rotate=0.0, roundness=0.0, thickness=0,
             invert=False, scale=1.0, xscale=1.0, yscale=1.0):
    pen = ngon(pos=pos, np=5, length=length, rotate=rotate+pi/10,
               roundness=roundness, invert=invert, scale=scale,
               xscale=xscale, yscale=yscale, thickness=thickness)
    return Polygon(pen)

def hexagon(pos=(0,0), length=1.0, rotate=0.0, roundness=0.0, thickness=0,
            invert=False, scale=1.0, xscale=1.0, yscale=1.0):
    hxg = ngon(pos=pos, np=6, length=length, rotate=rotate,
               roundness=roundness, invert=invert, scale=scale,
               xscale=xscale, yscale=yscale, thickness=thickness)
    return Polygon(hxg)

def star(pos=(0,0), radius=1.0, n=5, iradius=None, rotate=0.0, thickness=0.0,
         roundness=0.0, invert=False, scale=1.0, xscale=1.0, yscale=1.0):
    if iradius == None: iradius = radius*0.5
    if thickness == 0.0:
        pstar = Star(radius=radius, center=pos, beams=n, iradius=iradius)
        cp = pstar[0]
        cp.append(cp[0])
        cp.reverse() # Polygon Star goes around clockwise, so reverse to go CCW
        cp = rotatecp(cp, pos, rotate)
        if scale != 1.0: xscale = yscale = scale
        pp = Polygon(cp)
        if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
        if roundness > 0:
            cp = roundc(pp.contour(0), roundness=roundness, invert=invert)
            return Polygon(cp)
        else: return pp
    else:
        pp = sframe(pos=pos, radius=radius, iradius=iradius, rotate=rotate,
                    thickness=thickness, roundness=roundness, invert=invert,
                    scale=scale, xscale=xscale, yscale=yscale, n=n)
        return pp

def pointlist(pos=[], rotate=0.0, roundness=0.0, invert=False,
              scale=1.0, xscale=1.0, yscale=1.0, path=False):
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
    pp = Polygon(points)
    if len(pp) and rotate != 0.0: pp.rotate(rotate)
    if scale != 1.0: xscale = yscale = scale
    if xscale != 1.0 or yscale != 1.0: pp.scale(xscale,yscale)
    if roundness > 0:
        cp = roundc(pp.contour(0), roundness=roundness, invert=invert)
        pp = Polygon(cp)
    if path:
        if closed:
            return list(pp)
        else:
            return list(pp)[:-1]
    else:
        return pp

class text_info():
    pass

def text(pos=(0,0), text="", font=None, height=1.0, align='left', spacing=0.03, rotate=0.0,
         scale=1.0, xscale=1.0, yscale=1.0, thickness=None, vertical_spacing=None,
         info=False):
    # If info == True, the caller wants the additional info such as upper-left, etc.
    # In this case we return an instance of the class text_info, with the Polygon as
    # an attribute, text_info.Polygon. The main client of this extra info is the text object.
    if thickness is not None:
        raise AttributeError("Thickness is not allowed in a text shape.")
    if scale != 1.0: xscale = yscale = scale
    lines = text.split('\n')
    while lines[-1] == '\n': # strip off trailing newlines
        lines = lines[:-1]
    if font is None:
        font = "serif"
    font = describe.openFont(findFont(font))
    
    try:
        fonth = glyphquery.charHeight(font)
    except:
        fonth = 1000
    if fonth == 0: fonth = 1000
    
    try:
        desc = glyphquery.charDescent(font) # charDescent may not be present
        fontheight = fonth+desc
        fontscale = 1./fontheight
        descent = fontscale*desc
    except:
        descent = -0.3*height # approximate value
        fontheight = 0.7*fonth
        fontscale = 1.3/fontheight
        
    if vertical_spacing is None:
        vertical_spacing = height*fontscale*glyphquery.lineHeight(font)

    excludef_list = [("ITCEdscr")]
    excludec_list = [("FRSCRIPT", "A"), ("jokerman", "O"),
                    ("vivaldii", "O"), ("vivaldii", "Q"),
                    ("vivaldii", "R")]
    
    ptext = [] 
    widths = []
    width = 0.0
    starts = []
    start = 0.0

    for line in range(len(lines)):
        ptext.append(Polygon())
        bb = 0
        
        for newchar in lines[line]:
            if newchar == " ":
                try:
                    if a:
                        bx = a.boundingBox()
                        bba = bx[1]-bx[0]
                        bba = min(bba, 700)
                        bb += bba
                except:
                    pass
                continue
            n = glyphquery.glyphName(font, newchar)

            if n == ".notdef":
                print("The character '"+newchar+"' is not supported in the font "+font)
                continue
            
            g = glyph.Glyph(n)
            c = g.calculateContours(font)
            contours = []

            for contour in c:
                contours.append(glyph.decomposeOutline(contour))
                if len(contours[-1]) == 0: contours.pop()

            for contour in contours:
                pp = 0
                for i in range(len(contour)-1):
                    if (contour[i][0] == contour[i+1][0]) and (contour[i][1] == contour[i+1][1]):
                        contour.pop(i)
                        pp += 1
                        if i+pp >= len(contour)-1: break

            def lenctr(contour):
                totlen = 0
                for j in range(len(contour)-1):
                    totlen += (vector(contour[j])-vector(contour[j+1])).mag
                return totlen

            lc = len(contours)
            
            if lc >= 4:
                mli = 0
                maxl = 0
                lengths = []
                for i in range(len(contours)):
                    totlen = lenctr(contours[i])
                    lengths.append((totlen,i))
                    if totlen > maxl:
                        mli = i
                        maxl = totlen

                lengths.sort()
                lengths.reverse()
                ocontours = []
                for ll in lengths:
                    ocontours.append(contours[ll[1]])
                contours = ocontours

                indxf = -1
                indxc = -1
                if (mli > 0 and newchar != "%"):
                    try: indxf = excludef_list.index(self.font)
                    except: pass
                    if indxf == -1:
                        try: indxc = excludec_list.index((self.font, newchar))
                        except: pass
                    if (indxf == -1 and indxc == -1):
                        maxc = contours.pop(mli)
                        contours.insert(0, maxc)
                               
            a = Polygon(contours[0])
            for i in range(1,len(contours)):
                b = Polygon(contours[i])
                if a.covers(b): a = a - b
                elif b.covers(a): a = b - a
                else: a = a + b

            a.shift(bb - a.boundingBox()[0] ,0)
           
            ptext[line] += a
            
            bx = ptext[line].boundingBox()
            bb = bx[1] - bx[0] + spacing*fontheight

        newwidth = fontscale*height*(ptext[line].boundingBox()[1]-ptext[line].boundingBox()[0])
        widths.append(newwidth)
        if newwidth > width: width = newwidth
        
        ptext[line].scale(xscale*fontscale*height, yscale*fontscale*height, 0, 0)
        
    for line in range(len(lines)):
        if align == 'left':
            ptext[line].shift(-width/2,-line*vertical_spacing)
        elif align == 'right':
            ptext[line].shift(width/2-widths[line],-line*vertical_spacing)
        else:
            ptext[line].shift(-widths[line]/2,-line*vertical_spacing)
        ptext[line].shift(pos[0], pos[1])
        if rotate != 0.0: ptext[line].rotate(rotate)
        if line == 0:
            shape = ptext[0]
            upper_left = vector(ptext[line].boundingBox()[0],ptext[line].boundingBox()[3],0)
            upper_right = vector(ptext[line].boundingBox()[1],ptext[line].boundingBox()[3],0)
            lower_left = vector(ptext[line].boundingBox()[0],ptext[line].boundingBox()[2],0)
            lower_right = vector(ptext[line].boundingBox()[1],ptext[line].boundingBox()[2],0)
        else:
            shape += ptext[line]
            xleft = ptext[line].boundingBox()[0]
            xright = ptext[line].boundingBox()[1]
            y = ptext[line].boundingBox()[2]
            if xleft < upper_left.x:
                upper_left.x = xleft
            if xright > upper_right.x:
                upper_right.x = xright
            lower_left = vector(upper_left.x,ptext[line].boundingBox()[2],0)
            lower_right = vector(upper_right.x,ptext[line].boundingBox()[2],0)

    dy = vector(0,(upper_left.y-lower_left.y)/2-height,0)
    shape.shift(0,dy.y)
    if not info: return shape

    info = text_info()
    info.Polygon = shape
    info.upper_left = upper_left+dy
    info.upper_right = upper_right+dy
    info.lower_left = lower_left+dy
    info.lower_right = lower_right+dy
    if align == 'left':
        x = 0
    elif align == 'right':
        x = -width
    elif align == 'center':
        x = -0.5*width
    info.start = vector(x,0,0)+dy
    info.starts = []
    for line in range(len(lines)):
        if align == 'left':
            x = 0
        elif align == 'right':
            x = -widths[line]
        elif align == 'center':
            x = -0.5*widths[line]
        info.starts.append(vector(x,line*vertical_spacing,0)+dy)
    
    info.width = width
    info.widths = widths
    info.descent = descent
    info.vertical_spacing = vertical_spacing
    info.align = align
    info.height = height
    
    return info

##The following script has been developed and based on the
##Blender 235 script "Blender Mechanical Gears"
##developed in 2004 by Stefano <S68> Selleri,
##released under the Blender Artistic License (BAL).
##See www.blender.org.

####################################################################
#CREATES THE BASE INVOLUTE PROFILE
####################################################################
def ToothOutline(n=30, res=1, phi=20., radius=5.0, addendum=0.4, dedendum=0.5, fradius=0.1, bevel=0.05):
    TOOTHGEO = {
        'PitchRadius' : radius,
        'TeethN'      : n,
        'PressureAng' : phi,
        'Addendum'    : addendum,
        'Dedendum'    : dedendum,
        'Fillet'      : fradius,
        'Bevel'       : bevel,
        'Resolution'  : res,
        }   
    ####################################################################
    #Basic Math computations: Radii
    #
    R = {
        'Bottom'  : TOOTHGEO['PitchRadius'] - TOOTHGEO['Dedendum'] - TOOTHGEO['Fillet'],
        'Ded'     : TOOTHGEO['PitchRadius'] - TOOTHGEO['Dedendum'],
        'Base'    : TOOTHGEO['PitchRadius'] * cos(TOOTHGEO['PressureAng']*pi/180.0),
        'Bevel'   : TOOTHGEO['PitchRadius'] + TOOTHGEO['Addendum'] - TOOTHGEO['Bevel'],
        'Add'     : TOOTHGEO['PitchRadius'] + TOOTHGEO['Addendum']
    }

    ####################################################################
    #Basic Math computations: Angles
    #
    DiametralPitch = TOOTHGEO['TeethN']/(2*TOOTHGEO['PitchRadius'])
    ToothThickness = 1.5708/DiametralPitch
    CircularPitch  = pi / DiametralPitch

    U1 = sqrt((1-cos(TOOTHGEO['PressureAng']*pi/180.0))/
                   cos(TOOTHGEO['PressureAng']*pi/180.0))
    U2 = sqrt(R['Bevel']*R['Bevel']/(R['Ded']*R['Ded'])-1)

    ThetaA1 = atan((sin(U1)-U1*cos(U1))/(cos(U1)+U1*sin(U1)))
    ThetaA2 = atan((sin(U2)-U2*cos(U2))/(cos(U2)+U2*sin(U2)))
    ThetaA3 = ThetaA1 + ToothThickness/(TOOTHGEO['PitchRadius']*2.0)
    
    A = {
        'Theta0' : CircularPitch/(TOOTHGEO['PitchRadius']*2.0),
        'Theta1' : ThetaA3 + TOOTHGEO['Fillet']/R['Ded'],
        'Theta2' : ThetaA3,
        'Theta3' : ThetaA3 - ThetaA2,
        'Theta4' : ThetaA3 - ThetaA2 - TOOTHGEO['Bevel']/R['Add']
    }
    
    ####################################################################
    # Profiling
    #
    N = TOOTHGEO['Resolution']
    points  = []
    normals = []   
    # Top half bottom of tooth
    for i in range(2*N):
        th = (A['Theta1'] - A['Theta0'])*i/(2*N-1) + A['Theta0']              
        points.append ([R['Bottom']*cos(th),
                        R['Bottom']*sin(th)])
        normals.append([-cos(th),
                        -sin(th)])
        
    # Bottom Fillet
    xc = R['Ded']*cos(A['Theta1'])
    yc = R['Ded']*sin(A['Theta1'])
    Aw = pi/2.0 + A['Theta2'] - A['Theta1']
    for i in range(N):
        th = (Aw)*(i+1)/(N) + pi + A['Theta1']
        points.append ([xc + TOOTHGEO['Fillet']*cos(th),
                        yc + TOOTHGEO['Fillet']*sin(th)])
        normals.append([cos(th),
                        sin(th)])

    # Straight part
    for i in range(N):
        r = (R['Base']-R['Ded'])*(i+1)/(N) + R['Ded']              
        points.append ([r*cos(A['Theta2']),
                        r*sin(A['Theta2'])])
        normals.append([cos(A['Theta2']-pi/2.0),
                        sin(A['Theta2']-pi/2.0)])
    
    # Tooth Involute
    for i in range(3*N):
        r = (R['Bevel'] - R['Base'])*(i+1)/(3*N) + R['Base']
        u = sqrt(r*r/(R['Base']*R['Base'])-1)
        xp = R['Base']*(cos(u)+u*sin(u))
        yp = - R['Base']*(sin(u)-u*cos(u))
        points.append ([xp*cos(A['Theta2'])-yp*sin(A['Theta2']),
                        +xp*sin(A['Theta2'])+yp*cos(A['Theta2'])])
        normals.append([-sin(u),
                        -cos(u)])
        
    # Tooth Bevel
    auxth = -u 
    auxth = auxth + ThetaA3 + pi/2.0
    m     = tan(auxth)
    P0    = points[len(points)-1]
    rA    = TOOTHGEO['Bevel']/(1-cos(auxth-A['Theta4']))
    xc    = P0[0] - rA*cos(auxth)
    yc    = P0[1] - rA*sin(auxth)
    for i in range(N):
        th = (A['Theta4'] - auxth)*(i+1)/(N) + auxth              
        points.append ([xc + rA*cos(th),
                        yc +rA*sin(th)])
        normals.append([-cos(th),
                        -sin(th)])

    # Tooth Top
    P0    = points[len(points)-1]
    A['Theta4'] = atan (P0[1]/P0[0])
    Ra = sqrt(P0[0]*P0[0]+P0[1]*P0[1])
    for i in range(N):
        th = (-A['Theta4'])*(i+1)/(N) + A['Theta4']              
        points.append ([Ra*cos(th),
                        Ra*sin(th)])
        normals.append([-cos(th),
                        -sin(th)])

    # Mirrors this!
    N = len(points)
    for i in range(N-1):
        P = points[N-2-i]
        points.append([P[0],-P[1]])
        V = normals[N-2-i]
        normals.append([V[0],-V[1]])

    return points               # ,normals

####################################################################
#CREATES THE BASE RACK PROFILE
####################################################################
def RackOutline(n=30, res=1, phi=20., radius=5.0, addendum=0.4, dedendum=0.5, fradius=0.1, bevel=0.05):
    TOOTHGEO = {
        'PitchRadius' : radius,
        'TeethN'      : n,
        'PressureAng' : phi,
        'Addendum'    : addendum,
        'Dedendum'    : dedendum,
        'Fillet'      : fradius,
        'Bevel'       : bevel,
        'Resolution'  : res,
        }  
    ####################################################################
    #Basic Math computations: QUotes
    #
    X = {
        'Bottom'  :  - TOOTHGEO['Dedendum'] - TOOTHGEO['Fillet'],
        'Ded'     :  - TOOTHGEO['Dedendum'],
        'Bevel'   : TOOTHGEO['Addendum'] - TOOTHGEO['Bevel'],
        'Add'     : TOOTHGEO['Addendum']
    }

    ####################################################################
    #Basic Math computations: Angles
    #
    DiametralPitch = TOOTHGEO['TeethN']/(2*TOOTHGEO['PitchRadius'])
    ToothThickness = 1.5708/DiametralPitch
    CircularPitch  = pi / DiametralPitch

    Pa = TOOTHGEO['PressureAng']*pi/180.0
    
    yA1 = ToothThickness/2.0
    yA2 = (-X['Ded']+TOOTHGEO['Fillet']*sin(Pa))*tan(Pa)
    yA3 = TOOTHGEO['Fillet']*cos(Pa)

    A = {
        'y0' : CircularPitch/2.0,
        'y1' : yA1+yA2+yA3,
        'y2' : yA1+yA2,
        'y3' : yA1 -(X['Add']-TOOTHGEO['Bevel'])*tan(Pa),
        'y4' : yA1 -(X['Add']-TOOTHGEO['Bevel'])*tan(Pa)
                - cos(Pa)/(1-sin(Pa))*TOOTHGEO['Bevel']
    }

    ####################################################################
    # Profiling
    #
    N = TOOTHGEO['Resolution']
    points  = []
    normals = []
    ist = 0
    if fradius: ist = 1
    # Top half bottom of tooth
    for i in range(ist, 2*N):
        y = (A['y1'] - A['y0'])*i/(2*N-1) + A['y0']              
        points.append ([X['Bottom'],
                        y])
        normals.append([-1.0,
                        -0.0])
        
    # Bottom Fillet
    xc = X['Ded']
    yc = A['y1']
    Aw = pi/2.0 - Pa
    
    for i in range(N):
        th = (Aw)*(i+1)/(N) + pi
        points.append ([xc + TOOTHGEO['Fillet']*cos(th),
                        yc + TOOTHGEO['Fillet']*sin(th)])
        normals.append([cos(th),
                        sin(th)])

    # Straight part
    Xded = X['Ded'] - TOOTHGEO['Fillet']*sin(Pa)
    for i in range(4*N):
        x = (X['Bevel']-Xded)*(i+1)/(4*N) + Xded              
        points.append ([x,
                        yA1-tan(Pa)*x])
        normals.append([-sin(Pa),
                        -cos(Pa)])
    
    # Tooth Bevel
    rA    = TOOTHGEO['Bevel']/(1-sin(Pa))
    xc    =  X['Add'] - rA
    yc    =  A['y4']
    for i in range(N):
        th = (-pi/2.0+Pa)*(i+1)/(N) + pi/2.0-Pa
        points.append ([xc + rA*cos(th),
                        yc + rA*sin(th)])
        normals.append([-cos(th),
                        -sin(th)])

    # Tooth Top
    for i in range(N):
        y = -A['y4']*(i+1)/(N) + A['y4']
        points.append ([X['Add'],
                        y])
        normals.append([-1.0,
                        0.0])

    # Mirrors this!
    N = len(points)
    for i in range(N-1):
        P = points[N-2-i]
        points.append([P[0],-P[1]])
        V = normals[N-2-i]
        normals.append([V[0],-V[1]])

    return points               # ,normals

####################################################################
#CREATES THE BASE CROWN INVOLUTE 
####################################################################
def CrownOutline(n=30, res=1, phi=20., radius=5.0, addendum=0.4, dedendum=0.5, fradius=0.1, bevel=0.05):
    TOOTHGEO = {
        'PitchRadius' : radius,
        'TeethN'      : n,
        'PressureAng' : phi,
        'Addendum'    : addendum,
        'Dedendum'    : dedendum,
        'Fillet'      : fradius,
        'Bevel'       : bevel,
        'Resolution'  : res,
        }  
    ####################################################################
    #Basic Math computations: Radii
    #
    R = {
        'Bottom'  : TOOTHGEO['PitchRadius'] * cos(TOOTHGEO['PressureAng']*pi/180.0) ,
        'Base'    : TOOTHGEO['PitchRadius'] * cos(TOOTHGEO['PressureAng']*pi/180.0) + TOOTHGEO['Fillet'],
        'Ded'     : TOOTHGEO['PitchRadius'] + TOOTHGEO['Dedendum']
    }

    ####################################################################
    #Basic Math computations: Angles
    #
    DiametralPitch = TOOTHGEO['TeethN']/(2*TOOTHGEO['PitchRadius'])
    ToothThickness = 1.5708/DiametralPitch
    CircularPitch  = pi / DiametralPitch

    U1 = sqrt((1-cos(TOOTHGEO['PressureAng']*pi/180.0))/
                   cos(TOOTHGEO['PressureAng']*pi/180.0))
    U2 = sqrt(R['Ded']*R['Ded']/(R['Base']*R['Base'])-1)

    ThetaA1 = atan((sin(U1)-U1*cos(U1))/(cos(U1)+U1*sin(U1)))
    ThetaA2 = atan((sin(U2)-U2*cos(U2))/(cos(U2)+U2*sin(U2)))
    ThetaA3 = ThetaA1 + ToothThickness/(TOOTHGEO['PitchRadius']*2.0)
    
    A = {
        'Theta0' : CircularPitch/(TOOTHGEO['PitchRadius']*2.0),
        'Theta1' : (ThetaA3 + TOOTHGEO['Fillet']/R['Base']),
        'Theta2' : ThetaA3,
        'Theta3' : ThetaA3 - ThetaA2,
        'Theta4' : ThetaA3 - ThetaA2 - TOOTHGEO['Bevel']/R['Ded']
    }

    M = A['Theta0'] 
    A['Theta0'] = 0
    A['Theta1'] = A['Theta1']-M
    A['Theta2'] = A['Theta2']-M
    A['Theta3'] = A['Theta3']-M
    A['Theta4'] = A['Theta4']-M
    
    ####################################################################
    # Profiling
    #
    N = TOOTHGEO['Resolution']
    apoints  = []
    anormals = []   

    # Top half top of tooth
    for i in range(2*N):
        th = (A['Theta1'] - A['Theta0'])*i/(2*N-1) + A['Theta0']              
        apoints.append ([R['Bottom']*cos(th),
                        R['Bottom']*sin(th)])
        anormals.append([cos(th),
                        sin(th)])
        
    # Bottom Bevel
    xc = R['Base']*cos(A['Theta1'])
    yc = R['Base']*sin(A['Theta1'])
    Aw = pi/2.0 + A['Theta2'] - A['Theta1']
    for i in range(N):
        th = (Aw)*(i+1)/(N) + pi + A['Theta1']
        apoints.append ([xc + TOOTHGEO['Fillet']*cos(th),
                        yc + TOOTHGEO['Fillet']*sin(th)])
        anormals.append([-cos(th),
                        -sin(th)])

    # Tooth Involute
    for i in range(4*N):
        r = (R['Ded'] - R['Base'])*(i+1)/(4*N) + R['Base']
        u = sqrt(r*r/(R['Base']*R['Base'])-1)
        xp = R['Base']*(cos(u)+u*sin(u))
        yp = - R['Base']*(sin(u)-u*cos(u))
        apoints.append ([xp*cos(A['Theta2'])-yp*sin(A['Theta2']),
                        +xp*sin(A['Theta2'])+yp*cos(A['Theta2'])])
        anormals.append([sin(u),
                        cos(u)])
        
    # Tooth Bevel
    auxth = -u 
    auxth = auxth + ThetaA3 + pi/2.0
    m     = tan(auxth)
    P0    = apoints[len(apoints)-1]
    rA    = TOOTHGEO['Bevel']/(1-cos(auxth-A['Theta4']))
    xc    = P0[0] - rA*cos(auxth)
    yc    = P0[1] - rA*sin(auxth)
    for i in range(N):
        th = (A['Theta4'] - auxth)*(i+1)/(N) + auxth              
        apoints.append ([xc + rA*cos(th),
                        yc +rA*sin(th)])
        anormals.append([cos(th),
                        sin(th)])

    # Tooth Top
    P0    = apoints[len(apoints)-1]
    A['Theta4'] = atan (P0[1]/P0[0])
    Ra = sqrt(P0[0]*P0[0]+P0[1]*P0[1])
    for i in range(N):
        th = (-M - A['Theta4'])*(i+1)/(N) + A['Theta4']
        apoints.append ([Ra*cos(th),
                        Ra*sin(th)])
        anormals.append([cos(th),
                        sin(th)])
    points = []
    normals = []
    N = len(apoints)
    for i in range(N):
        points.append(apoints[N-1-i])
        normals.append(anormals[N-1-i])
        
    # Mirrors this!
    N = len(points)
    for i in range(N-1):
        P = points[N-2-i]
        points.append([P[0],-P[1]])
        V = normals[N-2-i]
        normals.append([V[0],-V[1]])

    return points           #,normals       process nromals later


def gear(pos=(0,0), n=20, radius=5, phi=20, addendum=0.4, dedendum=0.5,
         fradius=0.1, rotate=0, scale=1.0, internal=False, res=1, bevel=0):
        tooth = ToothOutline(n, res, phi, radius, addendum, dedendum,
                        fradius, bevel=0.0)
        if internal:
                itooth = []
                for p in tooth:
                        px = p[0]
                        py = p[1]
                        driro = sqrt(px*px +py*py) - radius
                        ir = radius - driro
                        ro = radius + driro
                        ix = (ir/ro)*px
                        iy = (ir/ro)*py
                        itooth.append((ix,iy))

                tooth = itooth
        gear = []
        for i in range(0, n):
            rotan = -i*2*pi/n
            rtooth = []
            for (x, y) in tooth:
                rx = x*cos(rotan) - y*sin(rotan) + pos[0]
                ry = x*sin(rotan) + y*cos(rotan) + pos[1]
                rtooth.append((rx,ry))
            gear.extend(rtooth)
        #gear.append(gear[0])
        pp =  Polygon(gear)
        if rotate != 0.0: pp.rotate(rotate)
        if scale != 1.0 : pp.scale(scale,scale)
        return pp


def rackGear(pos=(0,0), n=30, radius=5., phi=20., addendum=0.4, dedendum=0.5,
         fradius=0.1, rotate=0, scale=1.0, length=10*pi, res=1, bevel=0.05, depth=(0.4+0.6+0.1)):
        tooth = RackOutline(n, res, phi, radius, addendum, dedendum,
                        fradius, bevel)

        toothl = tooth[0][1] - tooth[-1][1]

        ntooth = int(length/toothl)
        flength = ntooth * toothl

        gear = []
        for i in range(0, ntooth):
            ntooth = []
            for (x, y) in tooth:
                nx = x + pos[0]
                ny = -i*toothl + y + pos[1]
                ntooth.append((nx,ny))
            gear.extend(ntooth)
        gear.append((gear[-1][0]-depth,gear[-1][1]))
        gear.append((gear[0][0]-depth,gear[0][1]))
        gear.append(gear[0])
        pp =  Polygon(gear)
        pp.shift(-pp.center()[0],-pp.center()[1])
        if rotate != 0.0: pp.rotate(rotate)
        if scale != 1.0 : pp.scale(scale,scale)
        return pp
