// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "cylinder.hpp"
#include "util/errors.hpp"
#include "util/displaylist.hpp"
#include "util/quadric.hpp"
#include "util/gl_enable.hpp"

namespace cvisual {

bool
cylinder::degenerate()
{
	return !visible || radius == 0.0 || axis.mag() == 0.0;
}

// TODO: This model currently uses three-deep glPushMatrix() to run.  It should
// be reduced.
static void
render_cylinder_model( size_t n_sides, size_t n_stacks = 1)
{
	quadric q;
	q.render_cylinder( 1.0, 1.0, n_sides, n_stacks);
	q.render_disk( 1.0, n_sides, 1, -1); // left end of cylinder
	gl_matrix_stackguard guard;
	glTranslatef( 1.0f, 0.0f, 0.0f);
	q.render_disk( 1.0, n_sides, 1, 1); // right end of cylinder
}

cylinder::cylinder()
{
}

cylinder::cylinder( const cylinder& other)
	: axial( other)
{
}

cylinder::~cylinder()
{
}

void
cylinder::init_model(view& scene)
{
	if (!scene.cylinder_model[0].compiled()) {
		clear_gl_error();
		// The number of faces corrisponding to each level of detail.
		size_t n_faces[] = { 8, 16, 32, 64, 96, 188 };
		size_t n_stacks[] = {1, 1, 3, 6, 10, 20 };
		for (size_t i = 0; i < 6; ++i) {
			scene.cylinder_model[i].gl_compile_begin();
			render_cylinder_model( n_faces[i], n_stacks[i]);
			scene.cylinder_model[i].gl_compile_end();
		}
		check_gl_error();
	}
}

void
cylinder::set_length( double l)
{
	axis = axis.norm() * l;
}

double
cylinder::get_length()
{
	return axis.mag();
}

void
cylinder::gl_pick_render( view& scene)
{
	if (degenerate())
		return;
	init_model(scene);

	size_t lod = 2;
	clear_gl_error();

	gl_matrix_stackguard guard;
	const double length = axis.mag();
	model_world_transform( scene.gcf, vector( length, radius, radius ) ).gl_mult();

	scene.cylinder_model[lod].gl_render();
	check_gl_error();
}

void
cylinder::gl_render( view& scene)
{
	if (degenerate())
		return;
	init_model(scene);

	clear_gl_error();

	// See sphere::gl_render() for a description of the level of detail calc.
	double coverage = scene.pixel_coverage( pos, radius);
	int lod = 0;
	if (coverage < 0)
		lod = 5;
	else if (coverage < 10)
		lod = 0;
	else if (coverage < 25)
		lod = 1;
	else if (coverage < 50)
		lod = 2;
	else if (coverage < 196)
		lod = 3;
	else if (coverage < 400)
		lod = 4;
	else
		lod = 5;
	lod += scene.lod_adjust;
	if (lod < 0)
		lod = 0;
	else if (lod > 5)
		lod = 5;

	gl_matrix_stackguard guard;
	const double length = axis.mag();
	model_world_transform( scene.gcf, vector( length, radius, radius ) ).gl_mult();

	if (translucent()) {
		gl_enable cull_face( GL_CULL_FACE);
		color.gl_set(opacity);

		// Render the back half.
		glCullFace( GL_FRONT);
		scene.cylinder_model[lod].gl_render();

		// Render the front half.
		glCullFace( GL_BACK);
		scene.cylinder_model[lod].gl_render();
	}
	else {
		color.gl_set(opacity);
		scene.cylinder_model[lod].gl_render();
	}

	// Cleanup.
	check_gl_error();
}

void
cylinder::grow_extent( extent& e)
{
	if (degenerate())
		return;
	vector a = axis.norm();
	e.add_circle(pos, a, radius);
	e.add_circle(pos+axis, a, radius);
	e.add_body();
}

vector
cylinder::get_center() const
{
	return pos + axis*0.5;
}

PRIMITIVE_TYPEINFO_IMPL(cylinder)

} // !namespace cvisual
