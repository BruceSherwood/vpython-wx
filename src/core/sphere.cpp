// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "sphere.hpp"
#include "util/quadric.hpp"
#include "util/errors.hpp"
#include "util/icososphere.hpp"
#include "util/gl_enable.hpp"
#include <iostream>

#include <vector>

namespace cvisual {

sphere::sphere()
{
}

sphere::sphere( const sphere& other)
	: axial(other)
{
}

sphere::~sphere()
{
}

void
sphere::gl_pick_render( view& geometry)
{
	if (degenerate())
		return;
	//init_model();
	init_model(geometry);

	gl_matrix_stackguard guard;
	model_world_transform( geometry.gcf, get_scale() ).gl_mult();

	geometry.sphere_model[0].gl_render();
	//check_gl_error();
}

void
sphere::gl_render( view& geometry)
{
	if (degenerate())
		return;

	//init_model();
	init_model(geometry);
	
	// coverage is the radius of this sphere in pixels:
	double coverage = geometry.pixel_coverage( pos, radius);
	int lod = 0;
	
	if (coverage < 0) // Behind the camera, but still visible.
		lod = 4;
	else if (coverage < 30)
		lod = 0;
	else if (coverage < 100)
		lod = 1;
	else if (coverage < 500)
		lod = 2;
	else if (coverage < 5000)
		lod = 3;
	else
		lod = 4;

	lod += geometry.lod_adjust; // allow user to reduce level of detail
	if (lod > 5)
		lod = 5;
	else if (lod < 0)
		lod = 0;

	gl_matrix_stackguard guard;
	model_world_transform( geometry.gcf, get_scale() ).gl_mult();

	color.gl_set(opacity);

	if (translucent()) {
		// Spheres are convex, so we don't need to sort
		gl_enable cull_face( GL_CULL_FACE);

		// Render the back half (inside)
		glCullFace( GL_FRONT );
		geometry.sphere_model[lod].gl_render();

		// Render the front half (outside)
		glCullFace( GL_BACK );
		geometry.sphere_model[lod].gl_render();
	}
	else {
		// Render a simple sphere.
		geometry.sphere_model[lod].gl_render();
	}
}

void
sphere::grow_extent( extent& e)
{
	e.add_sphere( pos, radius);
	e.add_body();
}

void
sphere::init_model(view& scene)
{
	//if (lod_cache[0]) return;
	if (scene.sphere_model[0].compiled()) return;

	quadric sph;
	
	scene.sphere_model[0].gl_compile_begin();
	sph.render_sphere( 1.0, 13, 7);
	scene.sphere_model[0].gl_compile_end();

	scene.sphere_model[1].gl_compile_begin();
	sph.render_sphere( 1.0, 19, 11);
	scene.sphere_model[1].gl_compile_end();

	scene.sphere_model[2].gl_compile_begin();
	sph.render_sphere( 1.0, 35, 19);
	scene.sphere_model[2].gl_compile_end();

	scene.sphere_model[3].gl_compile_begin();
	sph.render_sphere( 1.0, 55, 29);
	scene.sphere_model[3].gl_compile_end();

	scene.sphere_model[4].gl_compile_begin();
	sph.render_sphere( 1.0, 70, 34);
	scene.sphere_model[4].gl_compile_end();

	// Only for the very largest bodies.
	scene.sphere_model[5].gl_compile_begin();
	sph.render_sphere( 1.0, 140, 69);
	scene.sphere_model[5].gl_compile_end();
	
	//check_gl_error();
}

vector
sphere::get_scale()
{
	return vector( radius, radius, radius);
}

bool
sphere::degenerate()
{
	return !visible || radius == 0.0;
}

void
sphere::get_material_matrix(const view&, tmatrix& out) { 
	out.translate( vector(.5,.5,.5) ); 
	vector scale = get_scale();
	out.scale( scale * (.5 / std::max(scale.x, std::max(scale.y, scale.z))) ); 
}

PRIMITIVE_TYPEINFO_IMPL(sphere)

} // !namespace cvisual
