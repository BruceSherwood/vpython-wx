import os.path
import os
import sys
from platform import platform


# If we are working on a development version of IDLE, we need to prepend the
# parent of this idlelib dir to sys.path.  Otherwise, importing idlelib gets
# the version installed with the Python used to call this module:
idlelib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, idlelib_dir)

if platform().lower().split('-')[0] in ('darwin','linux'):
    from visual_common import __file__ as fname
    docPath = os.path.join(os.path.split(os.path.split(fname)[0])[0],'visual','examples')
    os.chdir(docPath)

import vidle.PyShell
vidle.PyShell.main()
