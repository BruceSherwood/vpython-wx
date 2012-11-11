// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "ring.hpp"
#include "util/displaylist.hpp"
#include "util/errors.hpp"
#include "util/gl_enable.hpp"

#include <utility>
#include <boost/scoped_array.hpp>
using boost::scoped_array;

namespace cvisual {

bool
ring::degenerate()
{
	return radius == 0.0;
}

ring::ring()
	: thickness(0.0), model_rings(-1)
{
}

ring::~ring()
{
}

void
ring::set_thickness( double t)
{
	thickness = t;
}

double
ring::get_thickness()
{
	return thickness;
}

void
ring::gl_pick_render(view& scene)
{
	gl_render(scene);
}

void
ring::gl_render(view& scene)
{
	if (degenerate())
		return;
	// Level of detail estimation.  See sphere::gl_render().

	// The number of subdivisions around the hoop's radial direction.
	double band_coverage = (thickness ? scene.pixel_coverage( pos, thickness)
			: scene.pixel_coverage(pos, radius*0.1));
	if (band_coverage<0) band_coverage = 1000;
	int bands = static_cast<int>( sqrt(band_coverage * 4.0) );
	bands = clamp( 4, bands, 40);
	// The number of subdivions around the hoop's tangential direction.
	double ring_coverage = scene.pixel_coverage( pos, radius);
	if (ring_coverage<0) ring_coverage = 1000;
	int rings = static_cast<int>( sqrt(ring_coverage * 4.0) );
	rings = clamp( 4, rings, 80);

	if (model_rings != rings || model_bands != bands || model_radius != radius || model_thickness != thickness) {
		model_rings = rings; model_bands = bands; model_radius = radius; model_thickness = thickness;
		create_model( rings, bands, model );
	}

	clear_gl_error();
	{
		gl_enable_client vertex_array( GL_VERTEX_ARRAY);
		gl_enable_client normal_array( GL_NORMAL_ARRAY);

		gl_matrix_stackguard guard;
		model_world_transform( scene.gcf, vector(radius,radius,radius) ).gl_mult();

		color.gl_set(opacity);

		glVertexPointer( 3, GL_FLOAT, 0, &model.vertex_pos[0] );
		glNormalPointer( GL_FLOAT, 0, &model.vertex_normal[0] );
		glDrawElements( GL_TRIANGLES, model.indices.size(), GL_UNSIGNED_SHORT, &model.indices[0] );
	}

	check_gl_error();
	return;
}

void
ring::grow_extent( extent& world)
{
	if (degenerate())
		return;
	// TODO: Not perfectly accurate (a couple more circles would help)
	vector a = axis.norm();
	double t = thickness ? thickness : radius * .1;
	world.add_circle( pos, a, radius + t );
	world.add_circle( pos + a*t, a, radius );
	world.add_circle( pos - a*t, a, radius );
	world.add_body();
}

void
ring::create_model( int rings, int bands, class model& m )
{
	// In Visual 3, rendered thickness was (incorrectly) double what was documented.
	// The documentation said that thickness was the diameter of a cross section of
	// a solid part of the ring, but in fact ring.thickness was the radius of the
	// cross section. Presumably we have to maintain the incorrect Visual 3 behavior
	// and change the documentation.
	double scaled_radius = 1.0;
	double scaled_thickness = 0.2;
	if (thickness != 0.0) scaled_thickness = 2*thickness / radius;

	// First generate a circle of radius thickness in the xy plane
	if (bands > 80) throw std::logic_error("ring::create_model: More bands than expected.");
	vector circle[80];
	circle[0] = vector(0,scaled_thickness*0.5,0);
	tmatrix rotator = rotation( 2.0 * M_PI / bands, vector( 0,0,1 ), vector( 0,0,0 ) );
	for (int i = 1; i < bands; i ++)
		circle[i] = rotator * circle[i-1];

	m.vertex_pos.resize( rings * bands );
	m.vertex_normal.resize( rings * bands );
	fvertex* vertexes = &m.vertex_pos[0];
	fvertex* normals = &m.vertex_normal[0];

	// ... and then sweep it in a circle around the x axis
	vector radial = vector(0,1,0);
	int i=0;
	rotator = rotation( 2.0 * M_PI / rings, vector( 1,0,0 ), vector( 0,0,0 ) );
	for(int r=0; r<rings; r++) {
		for(int b=0; b<bands; b++, i++) {
			normals[i].x = circle[b].x;
			normals[i].y = radial.y * circle[b].y;
			normals[i].z = radial.z * circle[b].y;
			vertexes[i].x = normals[i].x;
			vertexes[i].y = normals[i].y + radial.y;
			vertexes[i].z = normals[i].z + radial.z;
		}
		radial = rotator * radial;
	}

	// Now generate triangle indices... could do this with triangle strips but I'm looking
	// ahead to next renderer design, where it would be nice to always use indexed tris
	m.indices.resize( rings*bands*6 );
	unsigned short *ind = &m.indices[0];
	i = 0;
	for(int r=0; r<rings; r++) {
		for(int b=0; b<bands; b++,i++,ind+=6) {
			ind[0] = i; ind[1] = i+bands; ind[2] = i+1;
			ind[3] = i+bands; ind[4] = i+bands+1; ind[5] = i+1;
		}
		ind[2-6] -= bands;
		ind[4-6] -= bands;
		ind[5-6] -= bands;
	}
	ind -= 6*bands;
	for(int b=0; b<bands; b++,ind+=6) {
		ind[1] -= rings*bands;
		ind[3] -= rings*bands;
		ind[4] -= rings*bands;
	}
}

void
ring::get_material_matrix(const view&, tmatrix& out) {
	out.translate( vector(.5,.5,.5) );
	out.scale( vector(radius,radius,radius) * (.5 / (radius+thickness)) );
}

PRIMITIVE_TYPEINFO_IMPL(ring)

} // !namespace cvisual
