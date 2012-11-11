# This file may be modified by the user (or site administrator, in a
# multi-user environment) to change settings or modify VPython to work
# around local configuration issues.

## Disabling shaders may be necessary on some systems where Visual fails
## to detect that the video hardware or drivers are incapable of competently
## rendering the shaders used to implement certain materials.  This will
## revert all objects to legacy rendering regardless of what material is
## chosen:

##from .ui import display
##display.enable_shaders = False

## Alternatively, it might be possible to fix some systems by disabling
## only some materials, remapping them to ones that work:

##from . import materials
##materials.rough = materials.diffuse\
