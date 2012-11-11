#ifndef VPYTHON_PYTHON_CURVE_HPP
#define VPYTHON_PYTHON_CURVE_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "util/displaylist.hpp"
#include "python/num_util.hpp"
#include "python/arrayprim.hpp"

namespace cvisual { namespace python {

using boost::python::list;
using boost::python::numeric::array;

class curve : public arrayprim_color
{
 protected:
	// The pos and color arrays are always overallocated to make appends
	// faster.  Whenever they are read from Python, we return a slice into the
	// array that starts at its beginning and runs up to the last used position
	// in the array.  This is simmilar to many implementations of std::vector<>.
	bool antialias;
	double radius;

	static const int MAX_SIDES = 20;
	size_t sides;
	int curve_slice[512];
	float curve_sc[2*MAX_SIDES];

	// Returns true if the object is single-colored.
	bool monochrome(float* tcolor, size_t pcount);

	virtual void outer_render(view&);
	virtual void gl_render(view&);
	virtual vector get_center() const;
	virtual void gl_pick_render(view&);
	virtual void grow_extent( extent&);
	void get_material_matrix( const view& v, tmatrix& out );

	// Returns true if the object is degenarate and should not be rendered.
 	bool degenerate() const;

 public:
	curve();

	inline bool get_antialias( void) { return antialias; }
	inline double get_radius( void) { return radius; }

	void set_antialias( bool);
	void set_radius( const double& r);

 private:
	bool adjust_colors( const view& scene, float* tcolor, size_t pcount);
	void thickline( const view&, double* spos, float* tcolor, size_t pcount, double scaled_radius);
};

} } // !namespace cvisual::python

#endif // !VPYTHON_PYTHON_CURVE_HPP
