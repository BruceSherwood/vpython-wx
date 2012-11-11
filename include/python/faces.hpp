#ifndef VPYTHON_PYTHON_FACES_HPP
#define VPYTHON_PYTHON_FACES_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "python/arrayprim.hpp"

#include <boost/python/object.hpp>

namespace cvisual { namespace python {

class faces : public arrayprim_color
{
 protected:
	arrayprim_array<double> normal; // An array of normal vectors for the faces.

	virtual void set_length(size_t);

	bool degenerate() const;
	virtual void gl_render(view&);
	virtual void gl_pick_render(view&);
	virtual vector get_center() const;
	virtual void grow_extent( extent&);
	virtual void get_material_matrix( const view&, tmatrix& );

 public:
	faces();

	// Add another vertex, normal, and color to the faces.
	void append_rgb( const vector&, const vector&, float red=-1, float green=-1, float blue=-1);
	void append( const vector&, const vector&, const rgb& );
	void append( const vector&, const vector& );
	void append( const vector& );

	// This routine was adapted from faces_heightfield.py.  It averages the normal
	// vectors at coincident vertices to smooth out boundaries between facets,
	// for normals that are within about 15 degrees of each other (cos > .95).
	// The intent is to smooth what should be smooth but preserve sharp corners.
	void smooth();
	void smooth_d(const float);

	void make_normals();
	void make_twosided();
	
	boost::python::object get_normal();
	void set_normal( const double_array& normal);
	void set_normal_v( const vector);
};

} } // !namespace cvisual::python

#endif // !defined VPYTHON_PYTHON_FACES_HPP
