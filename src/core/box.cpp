// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "box.hpp"
#include "util/errors.hpp"
#include "util/gl_enable.hpp"

namespace cvisual {

void
//box::init_model( displaylist& model, bool skip_right_face ) {
box::init_model( view& scene, bool skip_right_face ) {
	// Note that this model is also used by arrow!
	scene.box_model.gl_compile_begin();
	glEnable(GL_CULL_FACE);
	glBegin( GL_QUADS );

	const float s = 0.5;
	float vertices[6][4][3] = {
		{{ +s, +s, +s }, { +s, -s, +s }, { +s, -s, -s }, { +s, +s, -s }}, // Right face
		{{ -s, +s, -s }, { -s, -s, -s }, { -s, -s, +s }, { -s, +s, +s }}, // Left face
		{{ -s, -s, +s }, { -s, -s, -s }, { +s, -s, -s }, { +s, -s, +s }}, // Bottom face
		{{ -s, +s, -s }, { -s, +s, +s }, { +s, +s, +s }, { +s, +s, -s }}, // Top face
		{{ +s, +s, +s }, { -s, +s, +s }, { -s, -s, +s }, { +s, -s, +s }}, // Front face
		{{ -s, -s, -s }, { -s, +s, -s }, { +s, +s, -s }, { +s, -s, -s }}  // Back face
	};
	float normals[6][3] = {
		{ +1, 0, 0 }, { -1, 0, 0 }, { 0, -1, 0 }, { 0, +1, 0 }, { 0, 0, +1 }, { 0, 0, -1 }
	};
	// Draw inside (reverse winding and normals)
	for(int f=skip_right_face; f<6; f++) {
		glNormal3f( -normals[f][0], -normals[f][1], -normals[f][2] );
		for(int v=0; v<4; v++)
			glVertex3fv( vertices[f][3-v] );
	}
	// Draw outside
	for(int f=skip_right_face; f<6; f++) {
		glNormal3fv( normals[f] );
		for(int v=0; v<4; v++)
			glVertex3fv( vertices[f][v] );
	}
	glEnd();
	glDisable(GL_CULL_FACE);
	scene.box_model.gl_compile_end();
	check_gl_error();
}

void 
//box::gl_pick_render( const view& scene)
box::gl_pick_render( view& scene)
{
	gl_render(scene);
}

void 
//box::gl_render( const view& scene)
box::gl_render( view& scene)
{
	if (!scene.box_model.compiled()) init_model(scene, false);

	color.gl_set(opacity);

	gl_matrix_stackguard guard;
	apply_transform( scene );
	
	scene.box_model.gl_render();
	check_gl_error();
}

void 
box::grow_extent( extent& e)
{
	tmatrix tm = model_world_transform( 1.0, vector( axis.mag(), height, width ) * 0.5 );
	e.add_box( tm, vector(-1,-1,-1), vector(1,1,1) );
	e.add_body();
}

void
box::get_material_matrix(const view&, tmatrix& out) { 
	out.translate( vector(.5,.5,.5) );
	vector scale( axis.mag(), height, width );
	out.scale( scale * (1.0 / std::max(scale.x, std::max(scale.y, scale.z))) );
}

PRIMITIVE_TYPEINFO_IMPL(box)

} // !namespace cvisual
