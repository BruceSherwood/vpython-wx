from __future__ import print_function
from visual import *

s = sphere()
instruct = """
Try mouse and keyboard events,
and note the printed outputs.
"""
l = label(pos=s.pos, text=instruct)
redrawCount = 0

class RedrawCounter(object):
    redrawCount = 0

    def increment(self):
        self.redrawCount += 1

def handleMouseDown(evt, arbArg):
    print("Mouse down!" + repr(evt.pos) + ':' + repr(arbArg) + ':' + evt.event)

    if m.enabled:
        m.stop()
        print("keydown events are now disabled")
    else:
        m.start()
        print("keydown events are now enabled")

def handleMouseUp( evt ):
    print("Mouse up! " + evt.event)

def handleMouseClick( evt ):
    print("Mouse click!" + evt.event)

def handleKeyUp( evt ):
    print("The ", evt.key, "key has gone up:", evt.event)
    print('   evt.ctrl =', evt.ctrl, ', evt.alt =', evt.alt, ', evt.shift =', evt.shift)

def handleKeyDown( evt ):
    print("The ", evt.key, "key has gone down", evt.event)
    print('   evt.ctrl =', evt.ctrl, ', evt.alt =', evt.alt, ', evt.shift =', evt.shift)
    if evt.key == 'R':
        print("There have been", redraw.redrawCount, "redraws")

def handleMouseMove( evt, num ):
    print(evt)
    print(num)
    print("Mouse moved! pos=", repr(evt.pos), ":", evt.event)

redraw = RedrawCounter()

scene.bind('mousedown', handleMouseDown, scene)
scene.bind('mouseup', handleMouseUp)
scene.bind('click', handleMouseClick)

m = scene.bind('keydown', handleKeyDown)
scene.bind('keyup', handleKeyUp)
scene.bind('redraw', redraw.increment)
scene.bind('mousemove', handleMouseMove, 3.2)
