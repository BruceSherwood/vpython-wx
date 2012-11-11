// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "material.hpp"

namespace cvisual {

// TODO: tan_hfov_x and tan_hfov_y must be revisited in the face of
// nonuniform scaling.  It may be more appropriate to describe the viewing
// frustum in a different way entirely.

view::view( const vector n_forward, vector n_center, int n_width,
	int n_height, bool n_forward_changed,
	double n_gcf, vector n_gcfvec,
	bool n_gcf_changed, gl_extensions& glext)
	: forward( n_forward), center(n_center), view_width( n_width),
	view_height( n_height), forward_changed( n_forward_changed),
	gcf( n_gcf), gcfvec( n_gcfvec), gcf_changed( n_gcf_changed), lod_adjust(0),
	anaglyph(false), coloranaglyph(false), tan_hfov_x(0), tan_hfov_y(0),
	screen_objects( z_comparator( forward)), glext(glext),
	enable_shaders(true)
{
	for(int i=0; i<N_LIGHT_TYPES; i++)
		light_count[i] = 0;
}

void view::apply_frame_transform( const tmatrix& wft ) {
	camera = wft * camera;
	forward = wft.times_v( forward );
	center = wft * center;
	up = wft.times_v(up);
	screen_objects_t tso( (z_comparator(forward)) );
	screen_objects.swap( tso );
}

double
view::pixel_coverage( const vector& pos, double radius) const
{
	// The distance from the camera to this position, in the direction of the
	// camera.  This is the distance to the viewing plane that the coverage
	// circle lies in.

	double dist = (pos - camera).dot(forward);
	// Half of the width of the viewing plane at this distance.
	double apparent_hwidth = tan_hfov_x * dist;
	// The fraction of the apparent width covered by the coverage circle.
	double coverage_fraction = radius / apparent_hwidth;
	// Convert from fraction to pixels.
	return coverage_fraction * view_width;

}

renderable::renderable()
	: visible(true), opacity( 1.0 )
{
}

renderable::~renderable()
{
}

void
renderable::outer_render( view& v )
{
	rgb actual_color = color;
	if (v.anaglyph) {
		if (v.coloranaglyph)
			color = actual_color.desaturate();
		else
			color = actual_color.grayscale();
	}

	tmatrix material_matrix;
	get_material_matrix(v, material_matrix);
	apply_material use_mat( v, mat.get(), material_matrix );
	gl_render(v);

	if (v.anaglyph)
		color = actual_color;
}

void
renderable::gl_render( view&)
{
	return;
}

void
renderable::gl_pick_render( view&)
{
}

void
renderable::grow_extent( extent&)
{
	return;
}

void
renderable::set_material( shared_ptr<class material> m )
{
	mat = m;
}

shared_ptr<class material>
renderable::get_material() {
	return mat;
}

bool renderable::translucent() {
	return opacity != 1.0 || (mat && mat->get_translucent());
}

} // !namespace cvisual
