#include "rectangular.hpp"

namespace cvisual {

rectangular::rectangular()
	: width(1.0), height(1.0)
{
}

rectangular::rectangular( const rectangular& other)
	: primitive( other), width( other.width), height( other.height)
{
}

rectangular::~rectangular()
{
}

void
rectangular::set_length( double l)
{
	if (l < 0)
		throw std::runtime_error( "length cannot be negative");
	axis = axis.norm() * l;
}

double
rectangular::get_length()
{
	return axis.mag();
}

void
rectangular::set_height( double h)
{
	if (h < 0)
		throw std::runtime_error( "height cannot be negative");
	height = h;
}

double
rectangular::get_height()
{
	return height;
}

void
rectangular::set_width( double w)
{
	if (w < 0)
		throw std::runtime_error( "width cannot be negative");
	width = w;
}

double
rectangular::get_width()
{
	return width;
}

vector
rectangular::get_size()
{
	return vector(axis.mag(), height, width);
}

void
rectangular::set_size( const vector& s)
{
	if (s.x < 0)
		throw std::runtime_error( "length cannot be negative");
	if (s.y < 0)
		throw std::runtime_error( "height cannot be negative");
	if (s.z < 0)
		throw std::runtime_error( "width cannot be negative");
	axis = axis.norm() * s.x;
	height = s.y;
	width = s.z;
}

void
rectangular::apply_transform( const view& scene )
{
	// OpenGL needs to invert the modelview matrix to generate the normal matrix,
	//   so try not to make it singular:
	double min_scale = std::max( axis.mag(), std::max(height,width) ) * 1e-6;
	vector size( std::max(min_scale,axis.mag()),
				 std::max(min_scale,height),
			     std::max(min_scale,width) );

	model_world_transform( scene.gcf, size ).gl_mult();
}

} // !namespace cvisual
