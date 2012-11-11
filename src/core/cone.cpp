// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "cone.hpp"
#include "util/errors.hpp"
#include "util/displaylist.hpp"
#include "util/quadric.hpp"
#include "util/gl_enable.hpp"

#include <vector>

#if 0
#include <boost/scoped_array.hpp>
namespace cvisual {
	using boost::scoped_array;
}
#endif

namespace cvisual {

bool
cone::degenerate()
{
	return !visible || radius == 0.0 || axis.mag() == 0.0;
}

static void
render_cone_model( size_t n_sides, size_t n_stacks = 1)
{
#if 0
	// I've swapped out this algorithm for the GLU version, but I'm keeping it
	// around in case I need it for something else later. -JDB

	// A rotation matrix to generate the edge vertexes and normals.
	tmatrix rotator = rotation( -2 * M_PI / n_sides, vector( 1, 0, 0));
	// A buffer to hold the last calculated ring.
	scoped_array<vector> vertexes( new vector[n_sides+1]);
	// Another buffer to hold the normal vectors (it is only initialized once,
	// and not overwritten later).
	scoped_array<vector> normals( new vector[n_sides+1]);

	// Render the base and calculate the normals and the first ring of vertexes.
	vertexes[0] = vector(0, 1, 0);
	normals[0] = vector(1,1,0).norm();
	glBegin( GL_TRIANGLE_FAN);
	vector(-1, 0, 0).gl_normal();
	vector( 0, 0, 0).gl_render();
	for (size_t i = 0; i < n_sides; ++i) {
		normals[i+1] = rotator * normals[i];
		vertexes[i+1] = rotator * vertexes[i];
		vertexes[i].gl_render();
	}
	vertexes[n_sides].gl_render();
	glEnd();

	// Render the body of the cone.
	double steps[] = { 0.3, 0.25, 0.2, 0.15, 0.1 };
	for (size_t i = 0; i < 5; ++i) {
		glBegin( GL_TRIANGLE_STRIP);
		normals[0].gl_normal();
		vertexes[0].gl_render();
		vertexes[0] = vertexes[0] + vector(steps[i], -steps[i]);
		vertexes[0].gl_render();
		for (size_t i = 1; i < n_sides+1; ++i) {
			normals[i].gl_normal();
			vertexes[i].gl_render();
			vertexes[i] = rotator * vertexes[i-1];
			vertexes[i].gl_render();
		}
		glEnd();
	}
#else
	quadric q;
	q.render_cylinder( 1.0, 0.0, 1.0, n_sides, n_stacks);
	q.render_disk( 1.0, n_sides, n_stacks * 2, -1);
#endif
}

cone::cone()
{
}

void
cone::set_length( double l)
{
	axis = axis.norm() * l;
}

double
cone::get_length()
{
	return axis.mag();
}

void
cone::init_model(view& scene)
{
	if (!scene.cone_model[0].compiled()) {
		clear_gl_error();
		// The number of faces corrisponding to each level of detail.
		size_t n_faces[] = { 8, 16, 32, 46, 68, 90 };
		size_t n_stacks[] = { 1, 2, 4, 7, 10, 14 };
		for (size_t i = 0; i < 6; ++i) {
			scene.cone_model[i].gl_compile_begin();
			render_cone_model( n_faces[i], n_stacks[i]);
			scene.cone_model[i].gl_compile_end();
		}
		check_gl_error();
	}
}

void
cone::gl_pick_render(view& scene)
{
	if (degenerate())
		return;
	init_model(scene);

	size_t lod = 2;
	clear_gl_error();

	gl_matrix_stackguard guard;
	const double length = axis.mag();
	model_world_transform( scene.gcf, vector( length, radius, radius ) ).gl_mult();

	scene.cone_model[lod].gl_render();
	check_gl_error();
}

void
cone::gl_render(view& scene)
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
	else if (coverage < 30)
		lod = 1;
	else if (coverage < 90)
		lod = 2;
	else if (coverage < 250)
		lod = 3;
	else if (coverage < 450)
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

	color.gl_set(opacity);

	if (translucent()) {
		gl_enable cull_face( GL_CULL_FACE);

		// Render the back half.
		glCullFace( GL_FRONT);
		scene.cone_model[lod].gl_render();

		// Render the front half.
		glCullFace( GL_BACK);
		scene.cone_model[lod].gl_render();
	}
	else {
		scene.cone_model[lod].gl_render();
	}

	check_gl_error();
}

void
cone::grow_extent( extent& e)
{
	if (degenerate())
		return;
	e.add_circle( pos, axis.norm(), radius );
	e.add_point( pos + axis);
	e.add_body();
}

vector
cone::get_center() const
{
	return pos + axis/2.0;
}

PRIMITIVE_TYPEINFO_IMPL(cone)

} // !namespace cvisual
