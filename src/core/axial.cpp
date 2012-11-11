#include "axial.hpp"

namespace cvisual {
	
axial::axial()
	: radius(1.0)
{
}

axial::axial( const axial& other)
	: primitive( other), radius( other.radius)
{
}

axial::~axial()
{
}

void
axial::set_radius( double r)
{
	radius = r;
}

double
axial::get_radius()
{
	return radius;
}

void
axial::get_material_matrix(const view&, tmatrix& out) { 
	out.translate( vector(.0005,.5,.5) );
	vector scale( axis.mag(), radius, radius );
	out.scale( scale * (.999 / std::max(scale.x, scale.y*2)) );
	
	// Undo the rotation inside quadric::render_cylinder() and ::render_disk():
	out = out * rotation( +.5*M_PI, vector(0,1,0) );  // xxx performance
}

} // !namespace cvisual
