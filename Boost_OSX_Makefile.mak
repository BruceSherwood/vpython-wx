SHELL = /bin/sh
CXX = /usr/local/bin/g++-3.3.4

VPATH = $(srcdir) $(srcdir)/converter $(srcdir)/detail $(srcdir)/object

.DEFAULT = all

boostroot = ../boost_1_31_0
srcdir = $(boostroot)/libs/python/src
boostincdir = $(boostroot)

WARNINGFLAGS =
OPTIMIZEFLAGS = -DNDEBUG -fpic -O3
BOOSTFLAGS = -DBOOST_PYTHON_MAX_BASES=2 -DBOOST_PYTHON_SOURCE \
	-DBOOST_PYTHON_DYNAMIC_LIB

OBJS = aix_init_module.lo dict.lo errors.lo list.lo long.lo module.lo \
	numeric.lo object_operators.lo object_protocol.lo str.lo tuple.lo \
	arg_to_python_base.lo from_python.lo registry.lo \
	type_id.lo class.lo enum.lo function.lo inheritance.lo iterator.lo \
	life_support.lo pickle_support.lo builtin_converters.lo

pythonroot = /sw
pythonincdir = $(pythonroot)/include/python2.3

DEPS = $(subst .lo,.d, $(OBJS))

%.lo: %.cpp
	$(CXX) -ftemplate-depth-120 $(WARNINGFLAGS) $(OPTIMIZEFLAGS) $(BOOSTFLAGS) -I$(pythonincdir) -I$(boostincdir) -c -o $@ $<

%.d: %.cpp
	$(CXX) $(BOOSTFLAGS) -I$(pythonincdir) -I$(boostincdir) -MM -MF $@ -MT "$*.lo $@" $<

DYLIB_FLAGS = -v -dynamiclib -undefined suppress -flat_namespace -compatibility_version 1.31.0 \
	-current_version 1.31.0
libboost_python.dylib: $(OBJS)
	ld -dynamic -m -r -d -o libboost_python.lo $^
	$(CXX) $(DYLIB_FLAGS) -o $@ libboost_python.lo 
	rm -f libboost_python.lo
	
libboost_python.a: $(OBJS)
	ar rs $@ $^


all: libboost_python.dylib libboost_python.a
-include $(DEPS)
