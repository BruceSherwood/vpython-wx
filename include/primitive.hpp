#ifndef VPYTHON_SIMPLE_DISPLAYOBJECT_HPP
#define VPYTHON_SIMPLE_DISPLAYOBJECT_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "util/tmatrix.hpp"

#include <typeinfo>

#include <boost/python/object.hpp>

namespace cvisual {

using boost::python::object;

// All primitive subclasses should use this pair of macros to help with standard
// error messages.  This allows functions to use the exact name of a virtual class.
#define PRIMITIVE_TYPEINFO_DECL virtual const std::type_info& get_typeid() const
#define PRIMITIVE_TYPEINFO_IMPL(base) \
	const std::type_info& \
	base::get_typeid() const \
	{ return typeid(*this); }

class primitive : public renderable
{
 protected:
	// The position and orientation of the body in World space.
	shared_vector axis;
	shared_vector up;
	shared_vector pos;

	bool make_trail, trail_initialized, obj_initialized;
	boost::python::object primitive_object;

	// Returns a tmatrix that performs reorientation of the object from model
	// orientation to world (and view) orientation.
	tmatrix model_world_transform( double world_scale = 0.0, const vector& object_scale = vector(1,1,1) ) const;
 
	// Generate a displayobject at the origin, with up pointing along +y and
	// an axis = vector(1, 0, 0).
	primitive();
	primitive( const primitive& other);
	
	// See above for PRIMITIVE_TYPEINFO_DECL/IMPL.
	virtual const std::type_info& get_typeid() const;
	
	// Used when obtaining the center of the body.
	virtual vector get_center() const;
	
 public:
	virtual ~primitive();

	// Manually overload this member since the default arguments are variables.
    void rotate( double angle, const vector& axis, const vector& origin);

	void set_pos( const vector& n_pos);
	shared_vector& get_pos();
	
	void set_x( double x);
	double get_x();
	
	void set_y( double y);
	double get_y();
	
	void set_z( double z);
	double get_z();
	
	void set_axis( const vector& n_axis);
	shared_vector& get_axis();
	
	void set_up( const vector& n_up);
	shared_vector& get_up();
	
	void set_color( const rgb& n_color);
	rgb get_color();
	
	void set_red( float x);
	double get_red();
	
	void set_green( float x);
	double get_green();
	
	void set_blue( float x);
	double get_blue();
	
	void set_opacity( float x);
	double get_opacity();

	void set_make_trail( bool x);
	bool get_make_trail();

	void set_primitive_object( boost::python::object x);
	boost::python::object get_primitive_object();
};

} // !namespace cvisual

#endif // !defined VPYTHON_SIMPLE_DISPLAYOBJECT_HPP
