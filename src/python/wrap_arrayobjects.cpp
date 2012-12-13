// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

// This file currently requires 144 MB to compile (optimizing).

#include "python/curve.hpp"
#include "python/extrusion.hpp"
#include "python/faces.hpp"
#include "python/convex.hpp"
#include "python/points.hpp"

#include "python/num_util.hpp"
#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/detail/wrap_python.hpp>

// Python 2/3 compatibility
#ifndef PyString_Check
#define PyString_Check   PyUnicode_Check
#endif

namespace cvisual {

using python::double_array;

struct double_array_from_python {
	double_array_from_python() {
		boost::python::converter::registry::push_back(
				&convertible,
				&construct,
				boost::python::type_id< double_array >());
	}

	static void* convertible(PyObject* obj_ptr)
	{
		using namespace boost::python;

		// TODO: We are supposed to determine if construct will succeed.  But
		//   this is difficult and expensive for arbitrary sequences.  So we
		//   assume that anything that looks like a sequence will convert and
		//   throw an exception later.  This limits overload resolution, but
		//   most of our functions taking arrays have no overloads.
		// Legend has it that numpy arrays don't satisfy PySequence_Check so
		///  we check if len(x) succeeds.
		if ( PySequence_Size(obj_ptr) < 0 ) {
			PyErr_Clear();
			return NULL;
		}
		// Strings have length but definitely don't convert to double_array!
		if ( PyString_Check(obj_ptr) || PyUnicode_Check(obj_ptr) )
			return NULL;

		return obj_ptr;
	}

	static void construct(
		PyObject* _obj,
		boost::python::converter::rvalue_from_python_stage1_data* data)
	{
		using namespace boost::python;

		void* storage = (
			(boost::python::converter::rvalue_from_python_storage<double_array>*)
			data)->storage.bytes;

		Py_INCREF(_obj);
		PyObject* arr = PyArray_FromAny(_obj, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ENSUREARRAY|NPY_CONTIGUOUS|NPY_ALIGNED, NULL);
		if (!arr)
			throw std::invalid_argument("Object cannot be converted to array.");

		new (storage) double_array( handle<>(arr) );

		data->convertible = storage;
	}
};

void
wrap_arrayobjects()
{
	using namespace boost::python;

	double_array_from_python();

	{
	using python::curve;

	// TODO: the arrayprim inheritance hierarchy could be exposed here; for now I've left the duplication here
	// to make it easy to control exactly what goes in the API for each array primitive, but arguably they
	// should be as similar as possible!

	void (curve::*append_v_rgb_retain)( const vector&, const rgb&, int ) = &curve::append;
	void (curve::*append_v_retain)( const vector&, int ) = &curve::append;

	class_<curve, bases<renderable> >( "curve")
		.def( init<const curve&>())
		.add_property( "radius", &curve::get_radius, &curve::set_radius)  // AKA thickness.
		.def( "get_color", &curve::get_color)
		.def( "set_color", &curve::set_color)
		.def( "set_red", &curve::set_red_d)
		.def( "set_red", &curve::set_red)
		.def( "set_green", &curve::set_green_d)
		.def( "set_green", &curve::set_green)
		.def( "set_blue", &curve::set_blue_d)
		.def( "set_blue", &curve::set_blue)
		.def( "get_pos", &curve::get_pos)
		.def( "set_pos", &curve::set_pos)
		.def( "set_pos", &curve::set_pos_v)
		.def( "set_x", &curve::set_x_d)
		.def( "set_x", &curve::set_x)
		.def( "set_y", &curve::set_y_d)
		.def( "set_y", &curve::set_y)
		.def( "set_z", &curve::set_z_d)
		.def( "set_z", &curve::set_z)
		.def( "append", append_v_rgb_retain, ( arg("pos"), arg("color"), arg("retain")=-1 ) )
		.def( "append", append_v_retain, ( arg("pos"), arg("retain")=-1 ) )
		.def( "append", &curve::append_rgb,
				( arg("pos"), arg("red")=-1, arg("green")=-1, arg("blue")=-1, arg("retain")=-1 ) )
		;
	}
	{
	using python::extrusion;

	class_<extrusion, bases<renderable> >( "extrusion")
		.def( init<const extrusion&>())
		.def( "get_color", &extrusion::get_color)
		.def( "set_color", &extrusion::set_color)
		.def( "set_red", &extrusion::set_red_d)
		.def( "set_red", &extrusion::set_red)
		.def( "set_green", &extrusion::set_green_d)
		.def( "set_green", &extrusion::set_green)
		.def( "set_blue", &extrusion::set_blue_d)
		.def( "set_blue", &extrusion::set_blue)
		.def( "get_pos", &extrusion::get_pos)
		.def( "set_pos", &extrusion::set_pos)
		.def( "set_pos", &extrusion::set_pos_v)
		.def( "set_x", &extrusion::set_x_d)
		.def( "set_x", &extrusion::set_x)
		.def( "set_y", &extrusion::set_y_d)
		.def( "set_y", &extrusion::set_y)
		.def( "set_z", &extrusion::set_z_d)
		.def( "set_z", &extrusion::set_z)
		// There were unsolvable problems with rotate. See comments with intrude routine.
		//.def( "rotate", &extrusion::rotate, (arg("angle"), arg("axis"), arg("origin")))
		.add_property( "up",
			make_function(&extrusion::get_up, return_internal_reference<>()),
			&extrusion::set_up)
		.add_property( "first_normal", &extrusion::get_first_normal, &extrusion::set_first_normal)
		.add_property( "last_normal", &extrusion::get_last_normal, &extrusion::set_last_normal)
		.add_property( "show_start_face", &extrusion::get_show_start_face, &extrusion::set_show_start_face)
		.add_property( "show_end_face", &extrusion::get_show_end_face, &extrusion::set_show_end_face)
		.add_property( "start", &extrusion::get_start, &extrusion::set_start)
		.add_property( "end", &extrusion::get_end, &extrusion::set_end)
		.add_property( "smooth", &extrusion::get_smooth, &extrusion::set_smooth)
		.add_property( "twosided", &extrusion::get_twosided, &extrusion::set_twosided)
		.add_property( "initial_twist", &extrusion::get_initial_twist, &extrusion::set_initial_twist)
		.def( "get_twist", &extrusion::get_twist)
		.def( "set_twist", &extrusion::set_twist)
		.def( "set_twist", &extrusion::set_twist_d)
		.def( "get_scale", &extrusion::get_scale)
		.def( "set_scale", &extrusion::set_scale)
		.def( "set_scale", &extrusion::set_scale_d)
		.def( "set_xscale", &extrusion::set_xscale)
		.def( "set_yscale", &extrusion::set_yscale)
		.def( "set_xscale", &extrusion::set_xscale_d)
		.def( "set_yscale", &extrusion::set_yscale_d)
		.def( "set_contours", &extrusion::set_contours) // used by primitives.py to transfer 2D cross section info
		.def( "_faces_render", &extrusion::_faces_render) // obtain pos, normal, and color arrays for the extrusion
		.def( "append", &extrusion::appendpos_retain, (arg("pos"), arg("retain")=-1))
		.def( "append", &extrusion::appendpos_color_retain, (arg("pos"), arg("color"), arg("retain")=-1))
		.def( "append", &extrusion::appendpos_rgb_retain,
				( arg("pos"), arg("red")=-1, arg("green")=-1, arg("blue")=-1, arg("retain")=-1 ) )
		;
	}
	{
	using python::points;

	void (points::*pappend_v_r)( const vector&, const rgb&, int ) = &points::append;
	void (points::*pappend_v)( const vector&, int ) = &points::append;

	class_<points, bases<renderable> >( "points")
		.def( init<const points&>())
		.add_property( "size", &points::get_size, &points::set_size)
		.add_property( "shape", &points::get_points_shape, &points::set_points_shape)
		.add_property( "size_units", &points::get_size_units, &points::set_size_units)
		.def( "get_color", &points::get_color)
		// The order of set_color specifications matters.
		//.def( "set_color", &points::set_color_t)
		.def( "set_color", &points::set_color)
		.def( "set_red", &points::set_red_d)
		.def( "set_red", &points::set_red)
		.def( "set_green", &points::set_green_d)
		.def( "set_green", &points::set_green)
		.def( "set_blue", &points::set_blue_d)
		.def( "set_blue", &points::set_blue)
		.def( "get_pos", &points::get_pos)
		.def( "set_pos", &points::set_pos)
		.def( "set_pos", &points::set_pos_v)
		.def( "set_x", &points::set_x_d)
		.def( "set_x", &points::set_x)
		.def( "set_y", &points::set_y_d)
		.def( "set_y", &points::set_y)
		.def( "set_z", &points::set_z_d)
		.def( "set_z", &points::set_z)
		.def( "append", pappend_v_r, (arg("pos"), arg("color"), arg("retain")=-1))
		.def( "append", pappend_v, (arg("pos"), arg("retain")=-1))
		.def( "append", &points::append_rgb,
			(arg("pos"), arg("red")=-1, arg("green")=-1, arg("blue")=-1, arg("retain")=-1))
		;
	}
	{
	using python::faces;

	void (faces::* append_all_vectors)(const vector&, const vector&, const rgb&) = &faces::append;
	void (faces::* append_default_color)(const vector&, const vector&) = &faces::append;
	void (faces::* append_pos)(const vector&) = &faces::append;

	class_<faces, bases<renderable> >("faces")
		.def( init<const faces&>())
		.def( "get_pos", &faces::get_pos)
		.def( "set_pos", &faces::set_pos)
		.def( "get_normal", &faces::get_normal)
		.def( "set_normal", &faces::set_normal_v)
		.def( "set_normal", &faces::set_normal)
		.def( "get_color", &faces::get_color)
		.def( "set_color", &faces::set_color)
		//.def( "set_color", &faces::set_color_t)
		.def( "set_red", &faces::set_red_d)
		.def( "set_red", &faces::set_red)
		.def( "set_green", &faces::set_green_d)
		.def( "set_green", &faces::set_green)
		.def( "set_blue", &faces::set_blue_d)
		.def( "set_blue", &faces::set_blue)
		.def( "smooth", &faces::smooth,
			"Average normal vectors at coincident vertexes.")
		.def( "smooth", &faces::smooth_d,
			"Average normal vectors at coincident vertexes.")
		.def( "make_normals", &faces::make_normals,
			"Construct normal vectors perpendicular to all faces.")
		.def( "make_twosided", &faces::make_twosided,
			"Add a second side and corresponding normals to all faces.")
		.def( "append", &faces::append_rgb,
			(arg("pos"), arg("normal"), arg("red")=-1, arg("green")=-1, arg("blue")=-1))
		.def( "append", append_pos, ( arg("pos") ))
		.def( "append", append_default_color, ( arg("pos"), arg("normal") ))
		.def( "append", append_all_vectors, (arg("pos"), arg("normal"), arg("color")))
		;
	}
	{
	using python::convex;

	void (convex::* append_convex)(const vector&) = &convex::append;
	class_<convex, bases<renderable> >( "convex")
		.def( init<const convex&>())
		.def( "append", append_convex, (arg("pos")),
		 	"Append a point to the surface in O(n) time.")
		.add_property( "color", &convex::get_color, &convex::set_color)
		.def( "set_pos", &convex::set_pos)
		.def( "get_pos", &convex::get_pos)
		;
	}

}

} // !namespace cvisual
