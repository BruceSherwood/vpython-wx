// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms. 
// See the file authors.txt for a complete list of contributors.

#include "python/convex.hpp"
#include "python/slice.hpp"
#include "util/gl_enable.hpp"
#include "util/errors.hpp"

#include <boost/python/extract.hpp>
#include <boost/crc.hpp>

namespace cvisual { namespace python {

convex::jitter_table convex::jitter;

long
convex::checksum() const
{
	boost::crc_32_type engine;
	engine.process_block( pos.data(), pos.end() );
	return engine.checksum();
}

bool
convex::degenerate() const
{
	return count < 3;
}

void
convex::recalc()
{
	hull.clear();

	const double* pos_i = pos.data();
	// A face from the first, second, and third vectors.
	hull.push_back( face( vector(pos_i), vector(pos_i+3), vector(pos_i+3*2)));
	// The reverse face from the first, third, and second vectors.
	hull.push_back( face( vector(pos_i), vector(pos_i+3*2), vector(pos_i+3)));
	// The remainder of the possible faces.
	for (size_t i = 3; i < count; ++i) {
		add_point( i, vector(pos_i + i*3));
	}
	
	// Calculate extents
	min_extent = max_extent = vector( pos_i );
	for(size_t i=1; i<count; i++)
		for(size_t j=0; j<3; j++) {
			if (*pos_i < min_extent[j]) min_extent[j] = *pos_i;
			else if (*pos_i > max_extent[j]) max_extent[j] = *pos_i;
			pos_i++;
		}

	last_checksum = checksum();
}

void
convex::add_point( size_t n, vector pv)
{
	double m = pv.mag();
	pv.x += m * jitter.v[(n  ) & jitter.mask];
	pv.y += m * jitter.v[(n+1) & jitter.mask];
	pv.z += m * jitter.v[(n+2) & jitter.mask];

	std::vector<edge> hole;
	for (size_t f=0; f<hull.size(); ) {
		if ( hull[f].visible_from(pv) ) {
			// hull[f] is visible from pv.  We will never get here if pv is
			//   inside the hull.

			// add the edges to the hole.  If an edge is already in the hole,
			//   it is not on the boundary of the hole and is removed.
			for(int e=0; e<3; ++e) {
				edge E( hull[f].corner[e], hull[f].corner[(e+1)%3] );

				bool boundary = true;
				for(std::vector<edge>::iterator h = hole.begin(); h != hole.end(); ++h) {
					if (*h == E) {
						*h = hole.back();
						hole.pop_back();
						boundary = false;
						break;
					}
				}

				if (boundary) {
					hole.push_back(E);
				}
			}

			// remove hull[f]
			hull[f] = hull.back();
			hull.pop_back();
		}
		else
			f++;
	}

	// Now add the boundary of the hole to the hull.  If pv was inside
	//   the hull, the hole will be empty and nothing happens here.
	for (std::vector<edge>::const_iterator h = hole.begin(); h != hole.end(); ++h) {
		hull.push_back(face(h->v[0], h->v[1], pv));
	}
}

convex::convex()
	: last_checksum(0)
{
}

void convex::set_color( const rgb& n_color)
{
	color = n_color;
}

rgb convex::get_color()
{
	return color;
}

void
convex::gl_render(view& scene)
{
	if (degenerate())
		return;
	long check = checksum();
	if (check != last_checksum) {
		recalc();
		last_checksum = check;
	}

	glShadeModel(GL_FLAT);
	gl_enable cull_face( GL_CULL_FACE);
	color.gl_set(1.0);

	glBegin(GL_TRIANGLES);
	for (std::vector<face>::const_iterator f = hull.begin(); f != hull.end(); ++f) {
		f->normal.gl_normal();
		(f->corner[0] * scene.gcf).gl_render();
		(f->corner[1] * scene.gcf).gl_render();
		(f->corner[2] * scene.gcf).gl_render();
	}
	glEnd();
	glShadeModel( GL_SMOOTH);
}

vector
convex::get_center() const
{
	if (degenerate())
		return vector();

	vector ret;
	for (std::vector<face>::const_iterator f = hull.begin(); f != hull.end(); ++f) {
		ret += f->center;
	}
	ret /= hull.empty() ? 1 : hull.size();

	return ret;
}

void
convex::gl_pick_render(view& scene)
{
	gl_render( scene);
}

void
convex::grow_extent( extent& world)
{
	if (degenerate())
		return;

	long check = checksum();
	if (check != last_checksum) {
		recalc();
	}
	assert( hull.size() != 0);

	for (std::vector<face>::const_iterator f = hull.begin(); f != hull.end(); ++f) {
		world.add_point( f->corner[0]);
		world.add_point( f->corner[1]);
		world.add_point( f->corner[2]);
	}
	world.add_body();
}

void 
convex::get_material_matrix( const view& v, tmatrix& out ) {
	out.translate( vector(.5,.5,.5) );
	
	out.scale( vector(1,1,1) * (.999 / (v.gcf * std::max(max_extent.x-min_extent.x, std::max(max_extent.y-min_extent.y, max_extent.z-min_extent.z)))) );
	
	out.translate( -.5 * v.gcf * (min_extent + max_extent) );
}

} } // !namespace cvisual::python
