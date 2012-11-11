# This is a simple makefile that in Jan. 2010 on Ubuntu 9.10 was equivalent
# to the autoconf build machinery. This file is unlikely to be kept up to date.
# Its purpose is just to show the main features of a Visual build, without the
# complexity of the autoconf mechanisms. It deals only with the building and
# installing of cvisualmodule.so, not with installing the other components (help etc.)

# Place a copy of this make file in a directory parallel to vpython-core2.
# Be sure to specify what version of Python.
# In vpython-core2/gtk2, run "sudo make setup" to establish links that make
# it unnecessary to move cvisualmodule.so into the actual site-packages directory.

prefix = /usr
PYTHON_VERSION = 2.6
PYTHON_INCLUDES = -I/usr/include/python$(PYTHON_VERSION) \
   -I$(prefix)/lib/python$(PYTHON_VERSION)/site-packages/numpy/core/include
#SITE_PACKAGES = /usr/lib/python$(PYTHON_VERSION)/site-packages/
TOP = ../vpython-core2
DEPENDS = $(TOP)/dependencies
SRC = $(TOP)/src
VPATH = $(SRC)/gtk2 $(SRC)/core/util $(SRC)/core $(SRC)/python $(SRC)

CXX = g++

GTK_LIBS = -Wl,--export-dynamic -lgtkglextmm-x11-1.2 -lgdkglextmm-x11-1.2 \
   -lgtkglext-x11-1.0 -lgdkglext-x11-1.0 -lGLU -lGL -lXmu -lXt -lSM -lICE \
   -lpangox-1.0 -lX11 -lpangoft2-1.0 -lfontconfig -lglademm-2.4 -lgtkmm-2.4 \
   -lglade-2.0 -lgdkmm-2.4 -latkmm-1.6 -lpangomm-1.4 -lcairomm-1.0 -lglibmm-2.4 \
   -lsigc-2.0 -lgtk-x11-2.0 -lxml2 -lgdk-x11-2.0 -latk-1.0 -lgdk_pixbuf-2.0 \
   -lm -lpangocairo-1.0 -lpango-1.0 -lcairo -lgobject-2.0 -lgmodule-2.0 -ldl \
   -lglib-2.0 -lfreetype -lz  
GTK_CFLAGS = -I/usr/include/gtkglextmm-1.2 -I/usr/lib/gtkglextmm-1.2/include \
   -I/usr/include/gtkglext-1.0 -I/usr/include/gtkmm-2.4 -I/usr/lib/gtkmm-2.4/include \
   -I/usr/lib/gtkglext-1.0/include -I/usr/include/gdkmm-2.4 -I/usr/lib/gdkmm-2.4/include \
   -I/usr/include/giomm-2.4 \
   -I/usr/include/pangomm-1.4 -I/usr/include/gtk-2.0 -I/usr/lib/gtk-2.0/include \
   -I/usr/lib/pangomm-1.4/include \
   -I/usr/include/pango-1.0 -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include \
   -I/usr/include/cairo -I/usr/include/freetype2 -I/usr/include/libpng12 \
   -I/usr/include/pixman-1 -I/usr/include/glibmm-2.4 -I/usr/lib/glibmm-2.4/include \
   -I/usr/include/cairomm-1.0 -I/usr/include/sigc++-2.0 -I/usr/lib/sigc++-2.0/include \
   -I/usr/include/atk-1.0 -I/usr/include/atkmm-1.6 -I/usr/include/libglademm-2.4 \
   -I/usr/lib/libglademm-2.4/include -I/usr/include/libglade-2.0 -I/usr/include/libxml2 \
   -I$(TOP)/include/gtk2
GTHREAD_LIBS = -pthread -lgthread-2.0 -lrt -lglib-2.0  
GTHREAD_CFLAGS = -pthread -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include  

# 18814435 MB without -DNDEBUG
# 18744287 MB with -DNDEBUG

# Option flags for the compiler, constructed from the above.
CVISUAL_CPPFLAGS = $(PYTHON_INCLUDES) -DHAVE_CONFIG_H -DNDEBUG \
        -I$(TOP)/dependencies/threadpool/include -I$(TOP)/include \
	$(GTK_CFLAGS) $(GTHREAD_CFLAGS)
CVISUAL_CXXFLAGS = -g -O2 -ftemplate-depth-120 -fPIC -DPIC

#_FILTER_OUT = "-Wl,--export-dynamic"
#CVISUAL_LIBS += $(filter-out $(_FILTER_OUT), $(GTK_LIBS) \
#       $(GTHREAD_LIBS) -lpython$(PYTHON_VERSION) -lboost_python-mt -lboost_thread-mt -lboost_signals-mt)
#CVISUAL_LIBS += -lstdc++

CVISUAL_LIBS += $(GTK_LIBS) $(GTHREAD_LIBS) \
       -lpython$(PYTHON_VERSION) -lboost_python-mt -lboost_signals-mt -lboost_thread-mt -lstdc++

CVISUAL_OBJS = arrayprim.o arrow.o axial.o box.o cone.o cylinder.o display_kernel.o ellipsoid.o \
	frame.o label.o material.o mouse_manager.o mouseobject.o primitive.o pyramid.o \
	rectangular.o renderable.o ring.o sphere.o text.o \
	atomic_queue.o displaylist.o errors.o extent.o \
	gl_extensions.o gl_free.o icososphere.o light.o quadric.o \
	display.o font_renderer.o random_device.o rate.o render_surface.o timer.o \
	render_manager.o rgba.o shader_program.o texture.o tmatrix.o vector.o\
	convex.o curve.o cvisualmodule.o faces.o \
	num_util.o numeric_texture.o points.o slice.o \
	wrap_arrayobjects.o wrap_display_kernel.o wrap_primitive.o \
	wrap_rgba.o wrap_vector.o

%.o: %.cpp
	$(CXX) $(CVISUAL_CPPFLAGS) $(CVISUAL_CXXFLAGS) -MMD -MF $*.d -MT "$*.d $*.lo" -c -o $@ $<

cvisualmodule.so : $(CVISUAL_OBJS)
	$(CXX) -shared $(CVISUAL_OBJS) $(CVISUAL_LIBS) -o cvisualmodule.so

# -rpath /usr/lib/python$(PYTHON_VERSION)/site-packages/cvisualmodule.la
	
install : cvisualmodule.so
	cp cvisualmodule.so ../vpython-core2/site-packages/visual
#	sudo cp cvisual.so /Library/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/lib/python$(PYTHON_VERSION)/site-packages/
#	sudo cp cvisual.so /Library/Python/$(PYTHON_VERSION)/site-packages

clean :
	rm *.d
	rm *.o
	rm *.so

# This command to be run from the vpython-core2/src/mac
setup : 
	mkdir -p ../../../vpython-core2/../mbuild
	ln -sfh $(abspath makefile) ../../../mbuild/makefile
	ln -sfh $(abspath ../../site-packages/visual) $(SITE_PACKAGES)visual

-include *.d
