// This file uses 152 MB to compile (optimizing)

// Copyright (c) 2003, 2004, 2005 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "primitive.hpp"
#include "material.hpp"
#include "arrow.hpp"
#include "sphere.hpp"
#include "cylinder.hpp"
#include "cone.hpp"
#include "ring.hpp"
#include "rectangular.hpp"
#include "box.hpp"
#include "ellipsoid.hpp"
#include "pyramid.hpp"
#include "label.hpp"
#include "frame.hpp"
#include "light.hpp"
#include "python/numeric_texture.hpp"

#include "python/wrap_vector.hpp"

#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/utility.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/raw_function.hpp>

namespace cvisual {
using namespace boost::python;
using boost::noncopyable;

/* Unfortunately the signatures of the functions primitive.rotate( "angle", "axis")
 * and primitive.rotate( "angle", "origin") are identical to Boost.Python.  To
 * differentiate them, I am using this raw function to interpret the arguments.
 * Ick.
 */
template <typename Prim>
object
py_rotate( tuple args, dict kwargs)
{
    Prim* This = extract<Prim*>( args[0]);

    if (!kwargs.has_key("angle")) {
        // This exception is more useful than the keyerror exception below.
        throw std::invalid_argument(
            "primitive.rotate(): angle of rotation must be specified.");
    }

    double angle = extract<double>(kwargs["angle"]);

    // The rotation axis, which defaults to the body axis.
    vector r_axis;
    if (kwargs.has_key("axis"))
        r_axis = tovector(kwargs["axis"]);
    else
        r_axis = This->get_axis();

    // The rotation origin, which defaults to the body position.
    vector origin;
    if (kwargs.has_key("origin"))
        origin = tovector(kwargs["origin"]);
    else
        origin = This->get_pos();

    This->rotate( angle, r_axis, origin);
    return object();
}

struct textures_to_list
{
	static PyObject* convert(std::vector<shared_ptr<texture> > const& a)
	{
		using namespace boost::python;
		list result;

		for(size_t i=0; i < a.size(); i++)
			result.append( a[i] );

		return incref(result.ptr());
	}
};

// This is a "custom rvalue converter". See also: Boost.Python FAQ
struct textures_from_list
{
	typedef std::vector<shared_ptr<texture> > V;

	textures_from_list()
	{
	  boost::python::converter::registry::push_back(
		&convertible,
		&construct,
		boost::python::type_id< V >());
	}

	static void* convertible(PyObject* obj_ptr)
	{
	  using namespace boost::python;
	  return obj_ptr; // if the input object is convertible
	}

	static void construct(
	  PyObject* obj_ptr,
	  boost::python::converter::rvalue_from_python_stage1_data* data)
	{
		using namespace boost::python;
		void* storage = ((converter::rvalue_from_python_storage<V>*)data)->storage.bytes;
		new (storage) V();
		data->convertible = storage;
		V& result = *((V*)storage);

		list l = extract< list >( obj_ptr );
		result.resize( len(l) );
		for(size_t i=0; i < result.size(); i++)
			result[i] = extract< shared_ptr<texture> >( l[i] );
	}
};

void
wrap_primitive()
{
	class_<renderable, boost::noncopyable>( "renderable", no_init)
		.add_property( "material", &renderable::get_material, &renderable::set_material)
		;

	class_<primitive, bases<renderable>, noncopyable>(
			"primitive", no_init)
		.add_property( "pos",
			make_function(&primitive::get_pos,
				return_internal_reference<>()),
			&primitive::set_pos)
		.add_property( "x", &primitive::get_x, &primitive::set_x)
		.add_property( "y", &primitive::get_y, &primitive::set_y)
		.add_property( "z", &primitive::get_z, &primitive::set_z)
		.add_property( "axis",
			make_function(&primitive::get_axis, return_internal_reference<>()),
			&primitive::set_axis)
		.add_property( "up",
			make_function(&primitive::get_up, return_internal_reference<>()),
			&primitive::set_up)
		.add_property( "color", &primitive::get_color, &primitive::set_color)
		.add_property( "red", &primitive::get_red, &primitive::set_red)
		.add_property( "green", &primitive::get_green, &primitive::set_green)
		.add_property( "blue", &primitive::get_blue, &primitive::set_blue)
		.add_property( "opacity", &primitive::get_opacity, &primitive::set_opacity)
		.add_property( "make_trail", &primitive::get_make_trail, &primitive::set_make_trail)
		.add_property( "primitive_object", &primitive::get_primitive_object, &primitive::set_primitive_object)
		 .def( "rotate", raw_function( &py_rotate<primitive>))
		;

	class_<axial, bases<primitive>, noncopyable>( "axial", no_init)
		.add_property( "radius", &axial::get_radius, &axial::set_radius)
		;

	class_<rectangular, bases<primitive>, noncopyable>( "rectangular", no_init)
		.add_property( "length", &rectangular::get_length, &rectangular::set_length)
		.add_property( "width", &rectangular::get_width, &rectangular::set_width)
		.add_property( "height", &rectangular::get_height, &rectangular::set_height)
		.add_property( "size", &rectangular::get_size, &rectangular::set_size)
		;

	class_< arrow, bases<primitive>, noncopyable >("arrow")
		.def( init<const arrow&>())
		.add_property( "length", &arrow::get_length, &arrow::set_length)
		.add_property( "shaftwidth", &arrow::get_shaftwidth, &arrow::set_shaftwidth)
		.add_property( "headlength", &arrow::get_headlength, &arrow::set_headlength)
		.add_property( "headwidth", &arrow::get_headwidth, &arrow::set_headwidth)
		.add_property( "fixedwidth", &arrow::is_fixedwidth, &arrow::set_fixedwidth)
		;

	class_< sphere, bases<axial> >( "sphere")
		.def( init<const sphere&>())
		;

	class_< cylinder, bases<axial> >( "cylinder")
		.def( init<const cylinder&>())
		.add_property( "length", &cylinder::get_length, &cylinder::set_length)
		;

	class_< cone, bases<axial> >( "cone")
		.def( init<const cone&>())
		.add_property( "length", &cone::get_length, &cone::set_length)
		;


	class_< ring, bases<axial> >( "ring")
		.def( init<const ring&>())
		.add_property( "thickness", &ring::get_thickness, &ring::set_thickness)
		;

	class_< box, bases<rectangular> >( "box")
		.def( init<const box&>())
		;

	// Actually this inherits from sphere, but this avoids unwrapping the radius
	// member.
	class_< ellipsoid, bases<primitive> >( "ellipsoid")
		.def( init<const ellipsoid&>())
		.add_property( "width", &ellipsoid::get_width, &ellipsoid::set_width)
		.add_property( "height", &ellipsoid::get_height, &ellipsoid::set_height)
		.add_property( "length", &ellipsoid::get_length, &ellipsoid::set_length)
		.add_property( "size", &ellipsoid::get_size, &ellipsoid::set_size)
		;

	class_< pyramid, bases<rectangular> >( "pyramid")
		.def( init<const pyramid&>())
		;

	class_<label, bases<renderable> >( "label")
		.def( init<const label&>())
		.add_property( "color", &label::get_color, &label::set_color)
		.add_property( "red", &label::get_red, &label::set_red)
		.add_property( "green", &label::get_green, &label::set_green)
		.add_property( "blue", &label::get_blue, &label::set_blue)
		.add_property( "opacity", &label::get_opacity, &label::set_opacity)
		.add_property( "pos",
			make_function(&label::get_pos, return_internal_reference<>()),
			&label::set_pos)
		.add_property( "x", &label::get_x, &label::set_x)
		.add_property( "y", &label::get_y, &label::set_y)
		.add_property( "z", &label::get_z, &label::set_z)
		.add_property( "height", &label::get_font_size, &label::set_font_size)
		.add_property( "xoffset", &label::get_xoffset, &label::set_xoffset)
		.add_property( "yoffset", &label::get_yoffset, &label::set_yoffset)
		.add_property( "border", &label::get_border, &label::set_border)
		.add_property( "box", &label::has_box, &label::render_box)
		.add_property( "line", &label::has_line, &label::render_line)
		.add_property( "linecolor", &label::get_linecolor, &label::set_linecolor)
		.add_property( "background", &label::get_background, &label::set_background)
		.add_property( "font", &label::get_font_family, &label::set_font_family)
		.add_property( "space", &label::get_space, &label::set_space)
		.add_property( "text", &label::get_text, &label::set_text)
		.add_property( "primitive_object", &label::get_primitive_object, &label::set_primitive_object)
		.def( "set_bitmap", &label::set_bitmap)
		;

	class_<frame, bases<renderable> >( "frame")
		.def( init<const frame&>())
		.add_property( "objects", &frame::get_objects)
		.add_property( "pos",
			make_function(&frame::get_pos, return_internal_reference<>()),
			&frame::set_pos)
		.add_property( "x", &frame::get_x, &frame::set_x)
		.add_property( "y", &frame::get_y, &frame::set_y)
		.add_property( "z", &frame::get_z, &frame::set_z)
		.add_property( "axis",
			make_function(&frame::get_axis, return_internal_reference<>()),
			&frame::set_axis)
		.add_property( "up",
			make_function(&frame::get_up, return_internal_reference<>()),
			&frame::set_up)
        .def( "rotate", raw_function( &py_rotate<frame>))
		.def( "add_renderable", &frame::add_renderable)
		.def( "remove_renderable", &frame::remove_renderable)
		.def( "frame_to_world", &frame::frame_to_world)
		.def( "world_to_frame", &frame::world_to_frame)
		;

	using python::numeric_texture;
	class_<texture, noncopyable>( "texbase", no_init);
	class_<numeric_texture, shared_ptr<numeric_texture>, bases<texture>, noncopyable>( "texture")
		.add_property( "data", &numeric_texture::get_data, &numeric_texture::set_data)
		.add_property( "type", &numeric_texture::get_type, &numeric_texture::set_type)
		.add_property( "mipmap", &numeric_texture::is_mipmapped, &numeric_texture::set_mipmapped)
		.add_property( "interpolate", &numeric_texture::is_antialiased, &numeric_texture::set_antialias)
		.add_property( "clamp", &numeric_texture::get_clamp, &numeric_texture::set_clamp)
		;

	boost::python::to_python_converter< std::vector< shared_ptr<texture> >, textures_to_list>();
	textures_from_list();
	class_<material, shared_ptr<material>, noncopyable>( "material" )
		.add_property( "textures", &material::get_textures, &material::set_textures )
		.add_property( "shader", &material::get_shader, &material::set_shader )
		.add_property( "translucent", &material::get_translucent, &material::set_translucent )
		;

	class_<light, bases<renderable>, noncopyable>( "light", no_init )
		.add_property( "color",
			&light::get_color, &light::set_color);
	class_<distant_light, bases<light>, noncopyable>( "distant_light" )
		.def( init<const distant_light&>())
		.add_property( "direction",
			make_function( &distant_light::get_direction,
				return_internal_reference<>()),
			&distant_light::set_direction);
	class_<local_light, bases<light>, noncopyable>( "local_light" )
		.def( init<const local_light&>())
		.add_property( "pos",
			make_function( &local_light::get_pos,
				return_internal_reference<>()),
			&local_light::set_pos);
}

} // !namespace cvisual
