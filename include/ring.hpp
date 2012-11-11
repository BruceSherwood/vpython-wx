#ifndef VPYTHON_RING_HPP
#define VPYTHON_RING_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "axial.hpp"

#ifdef __GNUC__
# define NONNULL __attribute__((nonnull))
#else
# define NONNULL
#endif

namespace cvisual {

// This model representation is intended to be "sort of like" what a next generation
// two phase renderer would use.  Eventually, therefore, it should be replaced with
// the real thing.
struct fvertex {
	union {
		float v[3];
		struct { float x, y, z; };
	};
	fvertex() {}  // uninitialized!
	fvertex( const vector& v ) : x(v.x), y(v.y), z(v.z) {}
};

class model {
public:
	std::vector< unsigned short > indices;
	std::vector< fvertex > vertex_pos;
	std::vector< fvertex > vertex_normal;
};

class ring : public axial
{
 private:
	// The radius of the ring's body.  If not specified, it is set to 1/10 of
	// the radius of the body.
	double thickness;
	PRIMITIVE_TYPEINFO_DECL;
	bool degenerate();

	cvisual::model model;
	int model_rings, model_bands;
	double model_radius, model_thickness;

 public:
	ring();
	virtual ~ring();
	void set_thickness( double t);
	double get_thickness();

 protected:
	virtual void gl_pick_render(view&);
	virtual void gl_render(view&);
	virtual void grow_extent( extent&);
	void get_material_matrix(const view&, tmatrix& out);

	void create_model( int rings, int bands, class model& m );
};

} // !namespace cvisual

#endif
