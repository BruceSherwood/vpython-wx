// This file requires 182 MB to compile (optimizing).

// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "display_kernel.hpp"
#include "mouseobject.hpp"
#include "util/errors.hpp"
#include <boost/bind.hpp>
#include <boost/python/class.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/to_python_converter.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/args.hpp>
#include <boost/python/list.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/def.hpp>
#include <boost/python/manage_new_object.hpp>

namespace cvisual {

// The callback function that is invoked from the display class when
// shutting-down. Prior to Jan. 23, 2008, force_py_exit locked and posted
// a callback to exit, but this failed if the program was sitting
// at scene.mouse.getclick(). Simply calling exit directly seems
// to work fine.
static void
force_py_exit(void)
{
	VPYTHON_NOTE("Exiting");
	std::exit(0);
}

namespace py = boost::python;

template < class Seq >
struct container_to_tuple
{
	static PyObject* convert( const Seq& items)
	{
		PyObject* p = PyTuple_New( items.size() );
		int index = 0;
		for( typename Seq::const_iterator item = items.begin(); item != items.end(); ++item, ++index ) {
			py::object o( *item );
			Py_INCREF(o.ptr());
			PyTuple_SET_ITEM(p, index, o.ptr());
		}
		return p;
	}
};

class py_base_display_kernel : public display_kernel {};
// Not clear why we need py_base_display_kernel?
// See http://wiki.python.org/moin/boost.python/OverridableVirtualFunctions

// A display implemented in python (e.g. to use PyOpenGL or PyObjC)
class py_display_kernel : public py_base_display_kernel
{
 public:
	PyObject* self;

	py_display_kernel( PyObject* self ) : self(self) {}

	// Delegates key display_kernel virtual methods to Python
	virtual void activate( bool active ) { boost::python::call_method<void>( self, "_activate", active ); }

	// Utility methods for Python subclasses
	void report_mouse_state(py::object is_button_down,
							int old_x, int old_y,
							int new_x, int new_y,
							py::object shift_state )
	{
		int button_len = boost::python::len( is_button_down );
		boost::scoped_array<bool> buttons( new bool[button_len] );
		for(int b = 0; b<button_len; b++)
			buttons[b] = boost::python::extract<bool>( is_button_down[b] );

		int shift_len = boost::python::len( shift_state );
		boost::scoped_array<bool> shift( new bool[shift_len] );
		for(int b=0; b<shift_len; b++)
			shift[b] = boost::python::extract<bool>( shift_state[b] );

		mouse.report_mouse_state( button_len, &buttons[0],
								  old_x, old_y,
								  new_x, new_y,
								  shift_len, &shift[0] );
	}
};

namespace {

boost::python::object
get_buttons( const mousebase* This)
{
	std::string* ret = This->get_buttons();
	if (ret) {
		boost::python::object py_ret( *ret);
		delete ret;
		return py_ret;
	}
	else {
		return boost::python::object();
	}
}

template <bool (mousebase:: *f)(void) const>
boost::python::object
test_state( const mousebase* This)
{
	if ((This->*f)()) {
		return get_buttons(This);
	}
	else
		return boost::python::object();
}

using namespace boost::python;
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS( pick_overloads, display_kernel::pick,
	2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS( mousebase_project_partial_overloads,
	mousebase::project2, 1, 2)
} // !namespace (unnamed)

void
wrap_display_kernel(void)
{
	using boost::noncopyable;

	py::class_<display_kernel, noncopyable>( "_display_kernel", no_init)
		// Functions for the internal use of renderable and light classes.
		.def( "add_renderable", &display_kernel::add_renderable)
		.def( "remove_renderable", &display_kernel::remove_renderable)
		.add_property( "up",
			py::make_function( &display_kernel::get_up,
				py::return_internal_reference<>()),
			&display_kernel::set_up)
		.add_property( "forward",
			py::make_function( &display_kernel::get_forward,
				py::return_internal_reference<>()),
			&display_kernel::set_forward)
		.add_property( "scale",	&display_kernel::get_scale,
			&display_kernel::set_scale)
		.add_property( "center",
			py::make_function( &display_kernel::get_center,
				py::return_internal_reference<>()),
			&display_kernel::set_center)
		.add_property( "fov", &display_kernel::get_fov, &display_kernel::set_fov)
		.add_property( "stereodepth", &display_kernel::get_stereodepth, &display_kernel::set_stereodepth)
		.add_property( "lod", &display_kernel::get_lod, &display_kernel::set_lod)
		.add_property( "uniform", &display_kernel::is_uniform,
		 	&display_kernel::set_uniform)
		.add_property( "background", &display_kernel::get_background,
			&display_kernel::set_background)
		.add_property( "foreground", &display_kernel::get_foreground,
			&display_kernel::set_foreground)
		.add_property( "autoscale", &display_kernel::get_autoscale,
			&display_kernel::set_autoscale)
		.add_property( "autocenter", &display_kernel::get_autocenter,
			&display_kernel::set_autocenter)
		.add_property( "stereo", &display_kernel::get_stereomode,
			&display_kernel::set_stereomode)
		.add_property( "userspin", &display_kernel::spin_is_allowed,
			&display_kernel::allow_spin)
		.add_property( "userzoom", &display_kernel::zoom_is_allowed,
			&display_kernel::allow_zoom)
		.def( "info", &display_kernel::info)
		.add_property( "x", &display_kernel::get_x, &display_kernel::set_x)
		.add_property( "y", &display_kernel::get_y, &display_kernel::set_y)
		.add_property( "width", &display_kernel::get_width, &display_kernel::set_width)
		.add_property( "height", &display_kernel::get_height, &display_kernel::set_height)
		.add_property( "title", &display_kernel::get_title, &display_kernel::set_title)
		.add_property( "fullscreen", &display_kernel::is_fullscreen,
			&display_kernel::set_fullscreen)
		// Omit toolbar for now; not yet available on Windows or Mac
		//.add_property( "toolbar", &display_kernel::is_showing_toolbar,
		//	&display_kernel::set_show_toolbar)
		.add_property( "visible", &display_kernel::get_visible, &display_kernel::set_visible)
		.add_property( "exit", &display_kernel::get_exit, &display_kernel::set_exit)
		.add_property( "mouse", py::make_function(
			&display_kernel::get_mouse, py::return_internal_reference<>()))
		.def( "set_selected", &display_kernel::set_selected)
		.staticmethod( "set_selected")
		.def( "get_selected", &display_kernel::get_selected)
		.staticmethod( "get_selected")

		.def( "_set_ambient", &display_kernel::set_ambient_f)
		.def( "_set_ambient", &display_kernel::set_ambient)
		.def( "_get_ambient", &display_kernel::get_ambient)

		.def( "_set_range", &display_kernel::set_range_d)
		.def( "_set_range", &display_kernel::set_range)
		.def( "_get_range", &display_kernel::get_range)

		.def( "_get_objects", &display_kernel::get_objects)

		.def_readwrite( "enable_shaders", &display_kernel::enable_shaders)
		;

	class_<py_base_display_kernel, py_display_kernel, bases<display_kernel>, noncopyable>
			( "display_kernel")
		// Default implementations of key override methods
		// Functions for extending this type in Python.
		.def( "render_scene", &display_kernel::render_scene )
		.def( "report_window_resize", &display_kernel::report_window_resize )
		.def( "report_view_resize", &display_kernel::report_view_resize )
		.def( "report_mouse_state", &py_display_kernel::report_mouse_state )
		.def( "report_closed", &display_kernel::report_closed )
		.def( "pick", &display_kernel::pick, pick_overloads(
			py::args( "x", "y", "pixels")))
		;

	py::to_python_converter<
		std::vector<shared_ptr<renderable> >,
		container_to_tuple< std::vector<shared_ptr<renderable> > > >();

	class_<mousebase>( "clickbase", no_init)
		.def( "project", &mousebase::project2,
			mousebase_project_partial_overloads( args("normal", "point")))
		.def( "project", &mousebase::project1, args("normal", "d"),
			"project the mouse pointer to the plane specified by the normal "
			"vector 'normal' that is either:\n-'d' distance away from the origin"
			" ( click.project( normal=vector, d=scalar)), or\n-includes the "
			"point 'point' ( click.project( normal=vector, point=vector))\n"
			"or passes through the origin ( click.project( normal=vector)).")
		.add_property( "pos", &mousebase::get_pos)
		.add_property( "pick", &mousebase::get_pick)
		.add_property( "pickpos", &mousebase::get_pickpos)
		.add_property( "camera", &mousebase::get_camera)
		.add_property( "ray", &mousebase::get_ray)
		.add_property( "button", &get_buttons)
		.add_property( "press", &test_state<&mousebase::is_press>)
		.add_property( "release", &test_state<&mousebase::is_release>)
		.add_property( "click", &test_state<&mousebase::is_click>)
		.add_property( "drag", &test_state<&mousebase::is_drag>)
		.add_property( "drop", &test_state<&mousebase::is_drop>)
		.add_property( "shift", &mousebase::is_shift)
		.add_property( "alt", &mousebase::is_alt)
		.add_property( "ctrl", &mousebase::is_ctrl)
		;


	class_< event, boost::shared_ptr<event>, bases<mousebase>, noncopyable>
		( "click_object", "An event generated from mouse actions.", no_init)
		;

	class_< mouse_t, boost::shared_ptr<mouse_t>, bases<mousebase>, noncopyable>
		( "mouse_object", "This class provides access to the mouse.", no_init)
		.def( "getclick", &mouse_t::pop_click)
		.add_property( "clicked", &mouse_t::num_clicks)
		.def( "getevent", &mouse_t::pop_event)
		.add_property( "events", &mouse_t::num_events, &mouse_t::clear_events)
		.def( "peekevent", &mouse_t::peek_event)
		;

}

} // !namespace cvisual;
