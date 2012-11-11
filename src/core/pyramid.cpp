// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "pyramid.hpp"
#include "util/errors.hpp"
#include "util/gl_enable.hpp"

namespace cvisual {

PRIMITIVE_TYPEINFO_IMPL(pyramid)

void
pyramid::init_model(view& scene)
{
	// Note that this model is also used by arrow!
	scene.pyramid_model.gl_compile_begin();
	
	float vertices[][3] = {
		{0, .5, .5},
		{0,-.5, .5},
		{0,-.5,-.5},
		{0, .5,-.5},
		{1,  0,  0}
	};
	int triangle_indices[][3] = { 
		{3, 0, 4},  // top
		{1, 2, 4},  // bottom
		{0, 1, 4},  // front
		{3, 4, 2},  // back
		{0, 3, 2},  // left (base) 1
		{0, 2, 1},  // left (base) 2
	};
	float normals[][3] = { {1,2,0}, {1,-2,0}, {1,0,2}, {1,0,-2}, {-1,0,0}, {-1,0,0} };

	glEnable(GL_CULL_FACE);
	glBegin( GL_TRIANGLES);

	// Inside
	for(int f=0; f<6; f++) {
		glNormal3f( -normals[f][0], -normals[f][1], -normals[f][2] );
		for(int v=0; v<3; v++)
			glVertex3fv( vertices[ triangle_indices[f][2-v] ] );
	}

	// Outside
	for(int f=0; f<6; f++) {
		glNormal3fv( normals[f] );
		for(int v=0; v<3; v++)
			glVertex3fv( vertices[ triangle_indices[f][v] ] );
	}

	glEnd();
	glDisable(GL_CULL_FACE);

	scene.pyramid_model.gl_compile_end();
	check_gl_error();
}

void 
pyramid::gl_pick_render( view& scene)
{
	gl_render(scene);
}

void 
pyramid::gl_render( view& scene)
{
	if (!scene.pyramid_model.compiled()) init_model(scene);

	color.gl_set(opacity);

	gl_matrix_stackguard guard;
	apply_transform( scene );

	scene.pyramid_model.gl_render();
	check_gl_error();
}

void 
pyramid::grow_extent( extent& world_extent)
{
	tmatrix orient = model_world_transform();
	vector vwidth = orient * vector( 0, 0, width * 0.5);
	vector vheight = orient * vector( 0, height * 0.5, 0);
	world_extent.add_point( pos + axis);
	world_extent.add_point( pos + vwidth + vheight);
	world_extent.add_point( pos - vwidth + vheight);
	world_extent.add_point( pos + vwidth - vheight);
	world_extent.add_point( pos - vwidth - vheight);
	world_extent.add_body();
}

vector
pyramid::get_center() const
{
	return pos + axis * 0.33333333333333;
}

void 
pyramid::get_material_matrix( const view&, tmatrix& out )
{
	out.translate( vector(0,.5,.5) );
	vector scale( axis.mag(), height, width );
	out.scale( scale * (1.0 / std::max(scale.x, std::max(scale.y, scale.z))) );
}

} // !namespace cvisual
