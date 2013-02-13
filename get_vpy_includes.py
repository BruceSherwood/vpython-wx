from __future__ import print_function
"""
Search for packages and include paths to build visual python on linux systems.
"""
import subprocess

gtk_list = ['gtk2','gtk+']
mm_gl_list = ['gtkmm','gtkglext','gtkglextmm']

def get_installed():
    """
    try to use pkg-config to get the right values for cflags
    """
    installed = []

    for line in subprocess.check_output(['pkg-config','--list-all']).split('\n'):
        line_list = line.split()
        if len(line_list):
            for pkg in mm_gl_list:
                if pkg in line_list[0]:
                    installed.append(line_list[0])
                    break

            for pkg in gtk_list:
                if pkg in line_list[0]:
                    installed.append(line_list[0])
                    break

    not_found = []
    for pkg in mm_gl_list:
        found = False
        for inst in installed:
            if pkg in inst:
                found = True
                break

        if not found:
            not_found.append(pkg)

    found = False
    for pkg in gtk_list:
        if not found:
            for inst in installed:
                if pkg in inst:
                    found = True
                    break


    errors = []
    if not_found:
        errors.append("can't find packages installed:" + ','.join(not_found))

    if not found:
        errors.append("Can't find gtk[2/+] package: " + ','.join(gtk_list))

    if errors:
        raise RuntimeError(','.join(errors))

    return installed

def get_includes():
    installed = get_installed()
    includes = []

    cflags_dict = {}

    for inst in installed:
        for item in subprocess.check_output(['pkg-config','--cflags',inst]).split():
            if item:
                cflags_dict[item] = 1


    for k in cflags_dict.keys():
        includes.append(k.replace('-I/','/'))

    return includes

def get_libs():
    installed = get_installed()
    libs = []

    libs_dict = {}

    for inst in installed:
        for item in subprocess.check_output(['pkg-config','--libs-only-l',inst]).split():
            if item:
                libs_dict[item] = 1


    for k in libs_dict.keys():
        libs.append(k.replace('-l',''))

    return libs

if __name__=='__main__':
    print("looks like we need:")
    print(get_includes())
    print(get_libs())









