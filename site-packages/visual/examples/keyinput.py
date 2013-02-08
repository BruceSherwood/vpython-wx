from __future__ import print_function
from visual import *
prose = label() # initially blank text

def keyInput(evt):
    s = evt.key
    if len(s) == 1:
        prose.text += s # append new character
    elif ((s == 'backspace' or s == 'delete') and
            len(prose.text)) > 0:
        if evt.shift:
            prose.text = '' # erase all text
        else:
            prose.text = prose.text[:-1] # erase letter

scene.bind('keydown', keyInput)
