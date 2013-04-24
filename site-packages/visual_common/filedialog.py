from visual import *
import wx
import os

### This is how you pre-establish a file filter so that the dialog
### only shows the extension(s) you want it to.
##wildcard = "Python source (*.py)|*.py|"     \
##           "Compiled Python (*.pyc)|*.pyc|" \
##           "SPAM files (*.spam)|*.spam|"    \
##           "Egg file (*.egg)|*.egg|"        \
##           "All files (*.*)|*.*"
        
def get_file_list(wildcard=''):
    # If the directory is changed in the process of getting files, this
    # dialog will change the current working directory to the path chosen.
    if wildcard != '' and wildcard[0] == '.': wildcard = '*'+wildcard
    
    dlg = wx.FileDialog(
        None, message="Choose a set of files",
        defaultDir=os.getcwd(), 
        defaultFile="",
        wildcard=wildcard,
        style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR
        )

    if dlg.ShowModal() == wx.ID_OK:
        # This returns a Python list of files that were selected.
        return dlg.GetPaths()
    else:
        return []

    dlg.Destroy()

def get_file(wildcard=''):
    if wildcard != '' and wildcard[0] == '.': wildcard = '*'+wildcard
    # If the directory is changed in the process of getting files, this
    # dialog will change the current working directory to the path chosen.
    dlg = wx.FileDialog(
        None, message="Choose a file",
        defaultDir=os.getcwd(), 
        defaultFile="",
        wildcard=wildcard,
        style=wx.OPEN | wx.CHANGE_DIR
        )

    if dlg.ShowModal() == wx.ID_OK:
        path = dlg.GetPath()
        try:
            fd = open(path, 'rU')
            return fd
        except:
            raise ValueError('Cannot read file '+dlg.GetPath())
    else:
        return None

    dlg.Destroy()

def save_file(wildcard=''):
    if wildcard != '' and wildcard[0] == '.': wildcard = '*'+wildcard
    # If the directory is changed in the process of getting files, this
    # dialog will change the current working directory to the path chosen.
    dlg = wx.FileDialog(
        None, message="Save file as ...", defaultDir=os.getcwd(), 
        defaultFile="", wildcard=wildcard, style=wx.SAVE | wx.FD_OVERWRITE_PROMPT | wx.CHANGE_DIR
        )

    if dlg.ShowModal() == wx.ID_OK:
        path = dlg.GetPath()
        try:
            fd = open(path, 'w')
            return fd
        except:
            raise ValueError('Cannot write file '+dlg.GetPath())
    else:
        return None

    dlg.Destroy()
