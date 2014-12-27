// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "primitive.hpp"
#include "util/errors.hpp"

#include <boost/python/import.hpp>

#include <typeinfo>
#include <cmath>

//#ifndef _MSC_VER // This was needed when we used GTK2 on Windows
#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <sstream>

namespace cvisual {

tmatrix 
primitive::model_world_transform(double world_scale, const vector& object_scale) const
{
	// Performs scale, rotation, translation, and world scale (gcf) transforms in that order.
	// ret = world_scale o translation o rotation o scale
	// Note that with the default parameters, only the rotation transformation is returned!  Typical
	//   usage should be model_world_transform( scene.gcf, my_size );

	tmatrix ret;
	// A unit vector along the z_axis.
	vector z_axis = vector(0,0,1);
	if (std::fabs(axis.dot(up) / std::sqrt( up.mag2() * axis.mag2())) > 0.98) {
		// Then axis and up are in (nearly) the same direction: therefore,
		// try two other possible directions for the up vector.
		if (std::fabs(axis.norm().dot( vector(-1,0,0))) > 0.98)
			z_axis = axis.cross( vector(0,0,1)).norm();
		else
			z_axis = axis.cross( vector(-1,0,0)).norm();
	}
	else {
		z_axis = axis.cross( up).norm();
	}
	
	vector y_axis = z_axis.cross(axis).norm();
	vector x_axis = axis.norm();
	ret.x_column( x_axis );
	ret.y_column( y_axis );
	ret.z_column( z_axis );
	ret.w_column( pos * world_scale );
	ret.w_row();

	ret.scale( object_scale * world_scale );

	return ret;
}

// Oblong objects (e.g. cylinder) whose center is not at "pos" override primitive::get_center
vector
primitive::get_center() const
{
	return pos;
}

using boost::python::import;
using boost::python::object;

bool startup = true;
boost::python::object trail_update;

primitive::primitive()
	: axis(1,0,0), up(0,1,0), pos(0,0,0), trail_initialized(false), obj_initialized(false)
{
}

primitive::primitive( const primitive& other)
	: renderable( other), axis(other.axis), up(other.up), 
		pos(other.pos)
{
}

primitive::~primitive()
{
}

void
primitive::rotate( double angle, const vector& _axis, const vector& origin)
{
	tmatrix R = rotation( angle, _axis, origin);
	vector fake_up = up;
	if (!axis.cross( fake_up)) {
		fake_up = vector( 1,0,0);
		if (!axis.cross( fake_up))
			fake_up = vector( 0,1,0);
	}
    {
    	pos = R * pos;
    	axis = R.times_v(axis);
    	up = R.times_v(fake_up);
    }
}

void 
primitive::set_pos( const vector& n_pos)
{
	pos = n_pos;
	if (trail_initialized && make_trail) {
		if (obj_initialized) {
			trail_update(primitive_object);
		}
	}
}

shared_vector&
primitive::get_pos()
{
	return pos;
}

void
primitive::set_x( double x)
{
	pos.set_x( x);
	if (trail_initialized && make_trail) {
		if (obj_initialized) {
			trail_update(primitive_object);
		}
	}
}

double
primitive::get_x()
{
	return pos.x;
}

void
primitive::set_y( double y)
{
	pos.set_y( y);
	if (trail_initialized && make_trail) {
		if (obj_initialized) {
			trail_update(primitive_object);
		}
	}
}

double
primitive::get_y()
{
	return pos.y;
}

void
primitive::set_z( double z)
{
	pos.set_z( z);
	if (trail_initialized && make_trail) {
		if (obj_initialized) {
			trail_update(primitive_object);
		}
	}
}

double
primitive::get_z()
{
	return pos.z;
}

void 
primitive::set_axis( const vector& n_axis)
{
	vector a = axis.cross(n_axis);
	if (a.mag() == 0.0) {
		axis = n_axis;
	} else {
		double angle = n_axis.diff_angle(axis);
		axis = n_axis.mag()*axis.norm();
		rotate(angle, a, pos);
	}
}

shared_vector&
primitive::get_axis()
{
	return axis;
}

void 
primitive::set_up( const vector& n_up)
{
	up = n_up;
}

shared_vector&
primitive::get_up()
{
	return up;
}

void 
primitive::set_color( const rgb& n_color)
{
	color = n_color;
}

void
primitive::set_red( float r)
{
	color.red = r;
}

double
primitive::get_red()
{
	return color.red;
}

void
primitive::set_green( float g)
{
	color.green = g;
}

double
primitive::get_green()
{
	return color.green;
}

void
primitive::set_blue( float b)
{
	color.blue = b;
}

double
primitive::get_blue()
{
	return color.blue;
}

rgb
primitive::get_color()
{
	return color;
}

void
primitive::set_opacity( float a)
{
	opacity = a;
}

double
primitive::get_opacity()
{
	return opacity;
}

void
primitive::set_make_trail( bool t)
{
	if (t && !obj_initialized)
		throw std::runtime_error( "Can't set make_trail=True unless object was created with make_trail specified");
	if (startup) {
		trail_update = import("visual_common.primitives").attr("trail_update");
		startup = false;
	}
	make_trail = t;
	trail_initialized = true;
}

bool
primitive::get_make_trail()
{
	return make_trail;
}

void
primitive::set_primitive_object( boost::python::object obj)
{
	primitive_object = obj;
	obj_initialized = true;
}

boost::python::object
primitive::get_primitive_object()
{
	return primitive_object;
}

PRIMITIVE_TYPEINFO_IMPL(primitive)

} // !namespace cvisual
