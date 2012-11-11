// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/extent.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <float.h>
#define QNAN (std::numeric_limits<double>::quiet_NaN())

namespace cvisual {

extent_data::extent_data(double tan_hfov)
: mins(QNAN,QNAN,QNAN),
  maxs(QNAN,QNAN,QNAN),
  camera_z(0),
  buffer_depth(0)
{
	cot_hfov = 1.0 / tan_hfov;
	sin_hfov = sin( atan(tan_hfov) );
	cos_hfov = sqrt( 1 - sin_hfov*sin_hfov );
	invsin_hfov = 1.0 / sin_hfov;
}

bool extent_data::is_empty() const { return !(mins.x == mins.x); } //< return isnan(mins.x)

vector extent_data::get_center() const {
	if (is_empty()) return vector();
	return (mins + maxs) * 0.5;
}

void extent_data::get_near_and_far( const vector& forward, double& nearest, double& farthest ) const
{
	if (is_empty() || (mins == maxs)) {
		// The only way that this should happen is if the scene is empty.
		nearest = 1.0;
		farthest = 10.0;
		return;
	}
    double corners[] = {
       maxs.dot(forward), // front upper right
       vector( mins.x, mins.y, maxs.z).dot(forward), // front lower left
       vector( mins.x, maxs.y, maxs.z).dot(forward), // front upper left
       vector( maxs.x, mins.y, maxs.z).dot(forward), // front lower right
       vector( mins.x, maxs.y, mins.z).dot(forward), // back upper left
       vector( maxs.x, mins.y, mins.z).dot(forward), // back lower right
       vector( maxs.x, maxs.y, mins.z).dot(forward) // back upper right
    };
    nearest = farthest = mins.dot(forward); // back lower left
    for (size_t i = 0; i < 7; ++i) {
		if (corners[i] < nearest) {
        	nearest = corners[i];
        }
		if (corners[i] > farthest) {
			farthest = corners[i];
		}
	}
}

vector extent_data::get_range( vector center) const {
    if (is_empty()) return vector(0,0,0);

	return vector(
		std::max( fabs( center.x - mins.x), fabs( center.x - maxs.x)),
		std::max( fabs( center.y - mins.y), fabs( center.y - maxs.y)),
		std::max( fabs( center.z - mins.z), fabs( center.z - maxs.z)));
}

//////////////////////////////////

extent::extent( extent_data& data, const tmatrix& local_to_centered_world )
	: data(data), l_cw( local_to_centered_world ), frame_depth(0)
{
}

extent::extent( extent& parent, const tmatrix& local_to_parent )
	: data( parent.data ), frame_depth(parent.frame_depth+1)
{
	l_cw = parent.l_cw * local_to_parent;
}

extent::~extent() {
}

void
extent::add_point( vector point)
{
	point = l_cw * point;

	// std::min(a,NAN) is defined as (NAN<a)?NAN:a, which is a.  So these will select point on the
	//   first call!
	data.mins.x = std::min( point.x, data.mins.x);
	data.maxs.x = std::max( point.x, data.maxs.x);
	data.mins.y = std::min( point.y, data.mins.y);
	data.maxs.y = std::max( point.y, data.maxs.y);
	data.mins.z = std::min( point.z, data.mins.z);
	data.maxs.z = std::max( point.z, data.maxs.z);

	// TODO: it might be more elegant (and even faster) to compute extents along
	//   specified axes (normal to the sides of the view frustum) and then display_kernel
	//   could compute camera_z from that.  The downside is that taking the fabs(point.z) out
	//   means 4 axes are needed instead of 2.
	data.camera_z  = std::max( data.camera_z,
		std::max(fabs(point.x),fabs(point.y))*data.cot_hfov + fabs(point.z) );
}

void
extent::add_sphere( vector center, double radius)
{
	radius = fabs(radius); //<TODO: why?
	center = l_cw * center;

	data.mins.x = std::min( center.x - radius, data.mins.x );
	data.maxs.x = std::max( center.x + radius, data.maxs.x );
	data.mins.y = std::min( center.y - radius, data.mins.y );
	data.maxs.y = std::max( center.y + radius, data.maxs.y );
	data.mins.z = std::min( center.z - radius, data.mins.z );
	data.maxs.z = std::max( center.z + radius, data.maxs.z );

	data.camera_z  = std::max( data.camera_z,
		std::max(fabs(center.x),fabs(center.y))*data.cot_hfov
			+ fabs(center.z)
			+ radius * data.invsin_hfov );
}

void
extent::add_box( const tmatrix& fwt, const vector& a, const vector& b ) {
	add_point( fwt * a );
	add_point( fwt * vector(a.x,a.y,b.z) );
	add_point( fwt * vector(a.x,b.y,a.z) );
	add_point( fwt * vector(a.x,b.y,b.z) );
	add_point( fwt * vector(b.x,a.y,a.z) );
	add_point( fwt * vector(b.x,a.y,b.z) );
	add_point( fwt * vector(b.x,b.y,a.z) );
	add_point( fwt * b );
}

void extent::add_circle( const vector& center, const vector& normal, double r ) {
	vector c = l_cw * center;
	vector n = l_cw.times_v(normal);

	vector n2( n.x*n.x, n.y*n.y, n.z*n.z );
	vector r_proj( r*sqrt(1.0 - n2.x), r*sqrt(1.0 - n2.y), r*sqrt(1.0 - n2.z) );

	data.mins.x = std::min( c.x - r_proj.x, data.mins.x );
	data.maxs.x = std::max( c.x + r_proj.x, data.maxs.x );
	data.mins.y = std::min( c.y - r_proj.y, data.mins.y );
	data.maxs.y = std::max( c.y + r_proj.y, data.maxs.y );
	data.mins.z = std::min( c.z - r_proj.z, data.mins.z );
	data.maxs.z = std::max( c.z + r_proj.z, data.maxs.z );

	double nxd = n.z*data.sin_hfov - n.x*data.cos_hfov;
	double nyd = n.z*data.sin_hfov - n.y*data.cos_hfov;
	data.camera_z  = std::max( data.camera_z,
		fabs(c.x)*data.cot_hfov
			+ fabs(c.z)
			+ r * sqrt(1.0 - nxd*nxd) * data.invsin_hfov );
	data.camera_z  = std::max( data.camera_z,
		fabs(c.y)*data.cot_hfov
			+ fabs(c.z)
			+ r * sqrt(1.0 - nyd*nyd) * data.invsin_hfov );
}

void
extent::add_body()
{
	data.buffer_depth += 4 + frame_depth;
}

} // !namespace cvisual
