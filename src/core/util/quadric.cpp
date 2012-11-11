// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "wrap_gl.hpp"
#include "util/quadric.hpp"
#include "util/tmatrix.hpp"

namespace cvisual {

quadric::quadric()
	: q(0)
{
	q = gluNewQuadric();
	gluQuadricDrawStyle( q, GLU_FILL);
	gluQuadricNormals( q, GLU_SMOOTH);
	gluQuadricOrientation( q, GLU_OUTSIDE);
}

quadric::~quadric()
{
	gluDeleteQuadric( q);
}

void 
quadric::set_draw_style( drawing_style style)
{
	switch (style) {
	case POINT:
		gluQuadricDrawStyle( q, GLU_POINT);
		break;
	case LINE:
		gluQuadricDrawStyle( q, GLU_LINE);
		break;
	case FILL:
		gluQuadricDrawStyle( q, GLU_FILL);
		break;
	case SILHOUETTE:
		gluQuadricDrawStyle( q, GLU_SILHOUETTE);
		break;
	}
}

void 
quadric::set_normal_style( normal_style style)
{
	switch (style) {
	case NONE:
		gluQuadricNormals( q, GLU_NONE);
		break;
	case FLAT:
		gluQuadricNormals( q, GLU_FLAT);
		break;
	case SMOOTH:
		gluQuadricNormals( q, GLU_SMOOTH);
		break;
	}
}

void 
quadric::set_orientation( orientation side)
{
	if (side == OUTSIDE)
		gluQuadricOrientation( q, GLU_OUTSIDE);
	else 
		gluQuadricOrientation( q, GLU_INSIDE);
}

void
quadric::render_sphere( double radius, int slices, int stacks)
{
	gluSphere( q, radius, slices, stacks);
}

void
quadric::render_cylinder( double base_radius, double top_radius, double height,
		int slices, int stacks)
{
	// GLU orients cylinders along the +z axis, and they must be
	// reoriented along the +x axis for VPython's convention of rendering along
	// the "axis" vector.
	gl_matrix_stackguard guard;
	glRotatef( 90, 0, 1, 0);
	gluCylinder( q, base_radius, top_radius, height, slices, stacks);
}

void
quadric::render_cylinder( double radius, double height, int slices, int stacks)
{
	gl_matrix_stackguard guard;
	glRotatef( 90, 0, 1, 0);
	gluCylinder( q, radius, radius, height, slices, stacks);
}

void 
quadric::render_disk( double radius, int slices, int rings, GLfloat rotation)
{
	gl_matrix_stackguard guard;
	glRotatef( 90, 0, rotation, 0);
	gluDisk( q, 0.0, radius, slices, rings);
}

} // !namespace cvisual
