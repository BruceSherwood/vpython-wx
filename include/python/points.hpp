#ifndef VPYTHON_PYTYHON_POINTS_HPP
#define VPYTHON_PYTYHON_POINTS_HPP

#include "renderable.hpp"
#include "python/num_util.hpp"
#include "python/arrayprim.hpp"

namespace cvisual { namespace python {

class points : public arrayprim_color
{
 private:
	// Specifies whether or not the size of the points should scale with the
	// world or with the screen.
	enum { WORLD, PIXELS } size_units;
	
	// Specifies the shape of the point. Future candidates are triangles,
	// diamonds, etc. 
	enum { ROUND, SQUARE } points_shape;

	// The size of the points
	float size;
	
	bool degenerate() const;
	
	virtual void outer_render(view&);
	virtual void gl_render(view&);
	virtual vector get_center() const;
	virtual void gl_pick_render(view&);
	virtual void grow_extent( extent&);
	
 public:
	points();
	
	void set_points_shape( const std::string& n_type);
	std::string get_points_shape( void);
	
	void set_size( float r);
	inline float get_size( void) { return size; }
	
	void set_size_units( const std::string& n_type);
	std::string get_size_units( void);
};

} } // !namespace cvisual::python

#endif /*VPYTHON_PYTYHON_POINTS_HPP*/
