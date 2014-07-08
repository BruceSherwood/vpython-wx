// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include <boost/python/detail/wrap_python.hpp>
#include <boost/crc.hpp>

#include "util/errors.hpp"
#include "util/gl_enable.hpp"

#include "python/slice.hpp"
#include "python/curve.hpp"

#include <stdexcept>
#include <cassert>
#include <sstream>
#include <iostream>

// Recall that the default constructor for object() is a reference to None.

namespace cvisual { namespace python {

curve::curve()
	: antialias( true), radius(0.0), sides(4)
{
	for (size_t i=0; i<sides; i++) {
		curve_sc[i]  = (float) std::cos(i * 2 * M_PI / sides);
		curve_sc[i+sides] = (float) std::sin(i * 2 * M_PI / sides);
	}

	// curve_slice is a list of indices for picking out the correct vertices from
	// a list of vertices representing one side of a thick-line curve. The lower
	// indices (0-255) are used for all but one of the sides. The upper indices
	// (256-511) are used for the final side.
	for (int i=0; i<128; i++) {
		curve_slice[i*2]       = i*sides;
		curve_slice[i*2+1]     = i*sides + 1;
		curve_slice[i*2 + 256] = i*sides + (sides - 1);
		curve_slice[i*2 + 257] = i*sides;
	}
}

void
curve::set_radius( const double& radius)
{
	this->radius = radius;
}

void
curve::set_antialias( bool aa)
{
	this->antialias = aa;
}

bool
curve::degenerate() const
{
	return count < 2;
}

bool
curve::monochrome(float* tcolor, size_t pcount)
{
	rgb first_color( tcolor[0], tcolor[1], tcolor[2]);
	size_t nn;

	for(nn=0; nn<pcount; nn++)  {
		if (tcolor[nn*3] != first_color.red)
			return false;
		if (tcolor[nn*3+1] != first_color.green)
			return false;
		if (tcolor[nn*3+2] != first_color.blue)
			return false;
	}

	return true;
}

namespace {
// Determines if two values differ by more than frac of either one.
bool
eq_frac( double lhs, double rhs, double frac)
{
	if (lhs == rhs)
		return true;
	double diff = fabs(lhs - rhs);
	lhs = fabs(lhs);
	rhs = fabs(rhs);
	return lhs*frac > diff && rhs*frac > diff;
}
} // !namespace (unnamed)

vector
curve::get_center() const
{
	// TODO: Optimize this by only recomputing the center when the checksum of
	// the entire object has changed.
	// TODO: Only add the "optimization" if the checksum is actually faster than
	// computing the average value every time...
	if (degenerate())
		return vector();
	vector ret;
	const double* pos_i = pos.data();
	const double* pos_end = pos.end();
	while (pos_i < pos_end) {
		ret.x += pos_i[0];
		ret.y += pos_i[1];
		ret.z += pos_i[2];
		pos_i += 3;
	}
	ret /= count;
	return ret;
}

void
curve::gl_pick_render(view& scene)
{
	// Aack, I can't think of any obvious optimizations here.
	// But since Visual 3 didn't permit picking of curves, omit for now.
	// We can't afford it; serious impact on performance.
	//gl_render( scene);
}

void
curve::grow_extent( extent& world)
{
	if (degenerate())
		return;
	const double* pos_i = pos.data();
	const double* pos_end = pos.end();
	if (radius == 0.0)
		for ( ; pos_i < pos_end; pos_i += 3)
			world.add_point( vector(pos_i));
	else
		for ( ; pos_i < pos_end; pos_i += 3)
			world.add_sphere( vector(pos_i), radius);
	world.add_body();
}

bool
curve::adjust_colors( const view& scene, float* tcolor, size_t pcount)
{
	rgb rendered_color;
	bool mono = monochrome(tcolor, pcount);
	if (mono) {
		// We can get away without using a color array.
		rendered_color = rgb( tcolor[0], tcolor[1], tcolor[2]);
		if (scene.anaglyph) {
			if (scene.coloranaglyph)
				rendered_color.desaturate().gl_set(opacity);
			else
				rendered_color.grayscale().gl_set(opacity);
		}
		else
			rendered_color.gl_set(opacity);
	}
	else {
		glEnableClientState( GL_COLOR_ARRAY);
		if (scene.anaglyph) {
			// Must desaturate or grayscale the color.

			for (size_t i = 0; i < pcount; ++i) {
				rendered_color = rgb( tcolor[3*i], tcolor[3*i+1], tcolor[3*i+2]);
				if (scene.coloranaglyph)
					rendered_color = rendered_color.desaturate();
				else
					rendered_color = rendered_color.grayscale();
				tcolor[3*i] = rendered_color.red;
				tcolor[3*i+1] = rendered_color.green;
				tcolor[3*i+2] = rendered_color.blue;
			}
		}
	}
	return mono;
}

namespace {
template <typename T>
struct converter
{
	T data[3];
};
} // !namespace (anonymous)

void
curve::thickline( const view& scene, double* spos, float* tcolor, size_t pcount, double scaled_radius)
{
	float *cost = curve_sc;
	float *sint = cost + sides;

	vector lastA; // unit vector of previous segment

	if (pcount < 2) return;

	bool closed = vector(&spos[0]) == vector(&spos[(pcount-1)*3]);

	size_t vcount = pcount*2 - closed;  // The number of vertices along each edge of the curve
	std::vector<vector> projected( vcount*sides );
	std::vector<vector> normals( vcount*sides );
	std::vector<rgb> light( vcount*sides );

	// pos and color iterators
	const double* v_i = spos;
	const float* c_i = tcolor;
	size_t i = closed ? 0 : sides;
	bool mono = adjust_colors( scene, tcolor, pcount);

	// eliminate initial duplicate points
	vector start( &v_i[0] );
	size_t reduce = 0;
	for (size_t corner=0; corner < pcount; ++corner, v_i += 3, c_i += 3) {
		vector next( &v_i[3] );
		vector A = (next - start).norm();
		if (!A) {
			reduce += 1;
			continue;
		}
		pcount -= reduce;
		break;
	}
	if (pcount < 2) return;

	for (size_t corner=0; corner < pcount; ++corner, v_i += 3, c_i += 3) {
		vector current( &v_i[0] );

		vector next, A, bisecting_plane_normal;
		double sectheta;
		if (corner != pcount-1) {
			next = vector( &v_i[3] ); // The next vector in spos
			A = (next - current).norm();
			if (!A) A = lastA;
			bisecting_plane_normal = (A + lastA).norm();
			if (!bisecting_plane_normal) {  //< Exactly 180 degree bend
				bisecting_plane_normal = vector(0,0,1).cross(A);
				if (!bisecting_plane_normal)
					bisecting_plane_normal = vector(0,1,0).cross(A);
			}
			sectheta = bisecting_plane_normal.dot( lastA );
			if (sectheta) sectheta = 1.0 / sectheta;
		}

		if (corner == 0) {
			vector y = vector(0,1,0);
			vector x = A.cross(y).norm();
			if (!x) {
				x = A.cross( vector(0, 0, 1)).norm();
			}
			y = x.cross(A).norm();

			if (!x || !y || x == y) {
				std::ostringstream msg;
				msg << "Degenerate curve case! please report the following "
					"information to visualpython-users@lists.sourceforge.net: ";
				msg << "current:" << current << " next:" << next
			 		<< " A:" << A << " x:" << x << " y:" << y
			 		<< std::endl;
			 	VPYTHON_WARNING( msg.str());
			}

			// scale radii
			x *= scaled_radius;
			y *= scaled_radius;

			for (size_t a=0; a < sides; a++) {
				vector rel = x*sint[a] + y*cost[a]; // first point is "up"

				normals[a+i] = rel.norm();
				projected[a+i] = current + rel;
				if (!mono) light[a+i] = rgb( c_i );

				if (!closed) {
					// Cap start of curve
					projected[a] = current;
					normals[a] = -A;
					if (!mono) light[a] = light[a+i];
				}
			}

			i += sides;
		} else {
			double Adot = A.dot(next - current);
			for (size_t a=0; a < sides; a++) {
				vector prev_start = projected[i+a-sides];
				vector rel = current - prev_start;
				double t = rel.dot(lastA);
				if (corner != pcount-1 && sectheta > 0.0) {
					double t1 = (rel.dot(bisecting_plane_normal)) * sectheta;
					t1 = std::max( t1, t - Adot );
					t = std::max( 0.0, std::min( t, t1 ) );
				}
				vector prev_end = prev_start + t*lastA;

				projected[i+a] = prev_end;
				normals[i+a] = normals[i+a-sides];
				if (!mono) light[i+a] = rgb( c_i );

				if (corner != pcount-1) {
					vector next_start = prev_end - 2*(prev_end-current).dot(bisecting_plane_normal)*bisecting_plane_normal;

					rel = next_start - current;

					projected[i+a+sides] = next_start;
					normals[i+a+sides] = (rel - A.dot(next_start-current)*A).norm();
					if (!mono) light[i+a+sides] = light[i+a];
				} else if (!closed) {
					// Cap end of curve
					for (size_t a=0; a < sides; a++) {
						projected[i+a+sides] = current;
						normals[i+a+sides] = lastA;
						if (!mono) light[i+a+sides] = light[a+i];
					}
				}
			}
			i += 2*sides;
		}
		lastA = A;
	}

	if (closed) {
		// Connect the end of the curve to the start... can be ugly because the basis has gotten
		//   twisted around!
		size_t i = (vcount - 1)*sides;
		for(size_t a=0; a<sides; a++) {
			projected[i+a] = projected[a];
			normals[i+a] = normals[a];
			if (!mono) light[i+a] = light[a];
		}
	}

	// Thick lines are often used to represent smooth curves, so we want
	// to smooth the normals at the joints.  But that can make a sharp corner
	// do odd things, so we smoothly disable the smoothing when the joint angle
	// is too big.  This is somewhat arbitrary but seems to work well.
	size_t prev_i = closed ? (vcount-1)*sides : 0;
	for( i = closed ? 0 : sides; i < vcount*sides; i += 2*sides ) {
		for(size_t a=0; a<sides; a++) {
			vector& n1 = normals[i+a];
			vector& n2 = normals[prev_i+a];
			double smooth_amount = (n1.dot(n2) - .65) * 4.0;
			smooth_amount = std::min(1.0, std::max(0.0, smooth_amount));
			if (smooth_amount) {
				vector n_smooth = (n1+n2).norm() * smooth_amount;
				n1 = n1 * (1-smooth_amount) + n_smooth;
				n2 = n2 * (1-smooth_amount) + n_smooth;
			}
		}
		prev_i = i + sides;
	}

	gl_enable_client vertex_arrays( GL_VERTEX_ARRAY);
	gl_enable_client normal_arrays( GL_NORMAL_ARRAY);
	if (!mono) {
		glEnableClientState( GL_COLOR_ARRAY);
	}

	int *ind = curve_slice;
	for (size_t a=0; a < sides; a++) {
		size_t ai = a;
		if (a == sides-1) {
			ind += 256; // upper portion of curve_slice indices, for the last side
			ai = 0;
		}

		// List all the vertices for the ai-th side of the thick line:
		for (size_t i = 0; i < vcount; i += 127u) {
			glVertexPointer(3, GL_DOUBLE, sizeof( vector), &projected[i*sides + ai].x);
			if (!mono)
				glColorPointer(3, GL_FLOAT, sizeof( rgb), &light[(i*sides + ai)].red );
			glNormalPointer( GL_DOUBLE, sizeof(vector), &normals[i*sides + ai].x);
			if (vcount-i < 128)
				glDrawElements(GL_TRIANGLE_STRIP, 2*(vcount-i), GL_UNSIGNED_INT, ind);
			else
				glDrawElements(GL_TRIANGLE_STRIP, 256u, GL_UNSIGNED_INT, ind);
		}
	}
	if (!mono)
		glDisableClientState( GL_COLOR_ARRAY);
}

void
curve::gl_render(view& scene)
{
	if (degenerate())
		return;
	const size_t true_size = count;
	// Set up the leading and trailing points for the joins.  See
	// glePolyCylinder() for details.  The intent is to create joins that are
	// perpendicular to the path at the last segment.  When the path appears
	// to be closed, it should be rendered that way on-screen.
	// The maximum number of points to display.
	const int LINE_LENGTH = 1000;
	// Data storage for the position and color data (plus room for 3 extra points)
	double spos[3*(LINE_LENGTH+3)];
	float tcolor[3*(LINE_LENGTH+3)]; // opacity not yet implemented for curves
	float fstep = (float)(count-1)/(float)(LINE_LENGTH-1);
	if (fstep < 1.0F) fstep = 1.0F;
	size_t iptr=0, iptr3, pcount=0;

	const double* p_i = pos.data();
	const double* c_i = color.data();

	// Choose which points to display
	for (float fptr=0.0; iptr < count && pcount < LINE_LENGTH; fptr += fstep, iptr = (int) (fptr+.5), ++pcount) {
		iptr3 = 3*iptr;
		spos[3*pcount] = p_i[iptr3];
		spos[3*pcount+1] = p_i[iptr3+1];
		spos[3*pcount+2] = p_i[iptr3+2];
		tcolor[3*pcount] = c_i[iptr3];
		tcolor[3*pcount+1] = c_i[iptr3+1];
		tcolor[3*pcount+2] = c_i[iptr3+2];
	}

	// Do scaling if necessary
	double scaled_radius = radius;
	if (scene.gcf != 1.0 || (scene.gcfvec[0] != scene.gcfvec[1])) {
		scaled_radius = radius*scene.gcfvec[0];
		for (size_t i = 0; i < pcount; ++i) {
			spos[3*i] *= scene.gcfvec[0];
			spos[3*i+1] *= scene.gcfvec[1];
			spos[3*i+2] *= scene.gcfvec[2];
		}
	}

	clear_gl_error();

	if (radius == 0.0) {
		glEnableClientState( GL_VERTEX_ARRAY);
		glDisable( GL_LIGHTING);
		// Assume monochrome.
		if (antialias) {
			glEnable( GL_LINE_SMOOTH);
		}

		glVertexPointer( 3, GL_DOUBLE, 0, spos);
		bool mono = adjust_colors( scene, tcolor, pcount);
		if (!mono) glColorPointer( 3, GL_FLOAT, 0, tcolor);
		glDrawArrays( GL_LINE_STRIP, 0, pcount);
		glDisableClientState( GL_VERTEX_ARRAY);
		glDisableClientState( GL_COLOR_ARRAY);
		glEnable( GL_LIGHTING);
		if (antialias) {
			glDisable( GL_LINE_SMOOTH);
		}
	}
	else {
		thickline( scene, spos, tcolor, pcount, scaled_radius);
	}

	check_gl_error();
}

void
curve::outer_render(view& v ) {
	if (radius)
		arrayprim::outer_render(v);
	else
		gl_render(v);  //< no materials
}

void
curve::get_material_matrix( const view& v, tmatrix& out ) {
	if (degenerate()) return;

	// TODO: note this code is identical to faces::get_material_matrix, except for considering radius

	// TODO: Add some caching for extent with grow_extent etc
	vector min_extent, max_extent;
	const double* pos_i = pos.data();
	const double* pos_end = pos.end();
	min_extent = max_extent = vector( pos_i ); pos_i += 3;
	while (pos_i < pos_end)
		for(int j=0; j<3; j++) {
			if (*pos_i < min_extent[j]) min_extent[j] = *pos_i;
			else if (*pos_i > max_extent[j]) max_extent[j] = *pos_i;
			pos_i++;
		}

	min_extent -= vector(radius,radius,radius);
	max_extent += vector(radius,radius,radius);

	out.translate( vector(.5,.5,.5) );
	out.scale( vector(1,1,1) * (.999 / (v.gcf * std::max(max_extent.x-min_extent.x, std::max(max_extent.y-min_extent.y, max_extent.z-min_extent.z)))) );
	out.translate( -.5 * v.gcf * (min_extent + max_extent) );
}

} } // !namespace cvisual::python
