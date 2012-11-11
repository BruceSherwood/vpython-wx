#include "python/points.hpp"
#include "python/num_util.hpp"
#include "python/slice.hpp"
#include "util/sorted_model.hpp"
#include "util/errors.hpp"
#include "util/gl_enable.hpp"

#include "wrap_gl.hpp"

#include <vector>
#include <sstream>
#include <algorithm>
#include <set>

namespace cvisual { namespace python {

using boost::python::make_tuple;
using boost::python::object;

points::points()
	: size_units(PIXELS), points_shape(ROUND), size( 5.0)
{
}

void points::set_size( float size) {
	this->size = size;
}

void points::set_points_shape( const std::string& n_type)
{
	if (n_type == "round") {
		points_shape = ROUND;
	}
	else if (n_type == "square") {
		points_shape = SQUARE;
	}
	else
		throw std::invalid_argument( "Unrecognized shape type");
}

std::string points::get_points_shape( void)
{
	switch (points_shape) {
		case ROUND:
			return "round";
		case SQUARE:
			return "square";
		default:
			return "";
	}
}

void points::set_size_units( const std::string& n_type)
{
	if (n_type == "pixels") {
		size_units = PIXELS;
	}
	else if (n_type == "world") {
		size_units = WORLD;
	}
	else
		throw std::invalid_argument( "Unrecognized coordinate type");
}

std::string points::get_size_units( void)
{
	switch (size_units) {
		case PIXELS:
			return "pixels";
		case WORLD:
			return "world";
		default:
			return "";
	}
}

bool
points::degenerate() const {
	return count == 0;
}

struct point_coord
{
	vector center;
	mutable rgb color;
	inline point_coord( const vector& p, const rgb& c)
		: center( p), color(c)
	{}
};

void
points::gl_render(view& scene)
{
	if (degenerate())
		return;

	std::vector<point_coord> translucent_points;
	typedef std::vector<point_coord>::iterator translucent_iterator;

	std::vector<point_coord> opaque_points;
	typedef std::vector<point_coord>::iterator opaque_iterator;

	const double* pos_i = pos.data();
	const double* pos_end = pos.end();

	const double* color_i = color.data();
	const double* color_end = color.end();

	// Currently points can not be translucent, so comment out all translucent code
	for ( ; pos_i < pos_end && color_i < color_end; pos_i += 3, color_i += 3) {
		opaque_points.push_back( point_coord( vector(pos_i), rgb(color_i)));
	}

	/*
	// First classify each point based on whether or not it is translucent
	if (points_shape == ROUND) { // Every point must be depth sorted
		for ( ; pos_i < pos_end && color_i < color_end; pos_i += 3, color_i += 3) {
			translucent_points.push_back( point_coord( vector(pos_i), rgb(color_i)));
		}
	}
	else { // Only translucent points need to be depth-sorted
		for ( ; pos_i < pos_end && color_i < color_end; pos_i += 3, color_i += 3) {
			if (0) // opacity not done
				translucent_points.push_back( point_coord( vector(pos_i), rgb(color_i)));
			else
				opaque_points.push_back( point_coord( vector(pos_i), rgb(color_i)));
		}
	}
	*/

	// Now conditionally apply transformations for gcf and anaglyph color
// Needs work
//	if (translucent_points.size())
//		renderable::color.opacity = 0.5;
	if (scene.gcf != 1.0 || (scene.gcfvec[0] != scene.gcfvec[1])) {
		for (opaque_iterator i = opaque_points.begin(); i != opaque_points.end(); ++i) {
			i->center = (i->center).scale(scene.gcfvec);
		}
		/*
		for (translucent_iterator i = translucent_points.begin(); i != translucent_points.end(); ++i) {
			i->center = (i->center).scale(scene.gcfvec);
		}
		*/
	}
	if (scene.anaglyph) {
		if (scene.coloranaglyph) {
			for (opaque_iterator i = opaque_points.begin(); i != opaque_points.end(); ++i) {
				i->color = i->color.desaturate();
			}
			/*
			for (translucent_iterator i = translucent_points.begin(); i != translucent_points.end(); ++i) {
				i->color = i->color.desaturate();
			}
			*/
		}
		else {
			for (opaque_iterator i = opaque_points.begin(); i != opaque_points.end(); ++i) {
				i->color = i->color.grayscale();
			}
			/*
			for (translucent_iterator i = translucent_points.begin(); i != translucent_points.end(); ++i) {
				i->color = i->color.grayscale();
			}
			*/
		}
	}
	/*
	// Sort the translucent points
	if (!translucent_points.empty()) {
		std::stable_sort( translucent_points.begin(), translucent_points.end(),
			face_z_comparator(scene.forward));
	}
	*/

	clear_gl_error();

	if (points_shape == ROUND)
		glEnable( GL_POINT_SMOOTH);

	if (size_units == WORLD && scene.glext.ARB_point_parameters) {
		// This is simpler and more robust than what was here before, but it's still
		// a little tacky and probably not perfectly general.  I'm not sure that it
		// should work with stereo frustums, but I can't find a case where it's
		// obviously wrong.
		// However, note that point attenuation (regardless of parameters) isn't a
		// correct perspective calculation, because it divides by distance, not by Z.
		// Points not at the center of the screen will be too small, particularly
		// at high fields of view.  This is in addition to the implementation limits
		// on point size, which will be a problem when points get too big or close.

		tmatrix proj; proj.gl_projection_get();  // Projection matrix

		vector p = (proj * vertex(.5,0,1,1)).project();  // eye coordinates .5,0,1 -> window coordinates

		// At an eye z of 1, a sphere of world-space diameter 1 is p.x * scene.view_width pixels wide,
		// so a sphere of world-space diameter (size*scene.gcf) is
		double point_radius_at_z_1 = size * scene.gcf * p.x * scene.view_width;

		//float attenuation_eqn[] =  { 0.0f, 0.0f, 1.0f / (float)(point_radius_at_z_1*point_radius_at_z_1) };
		//scene.glext.glPointParameterfvARB( GL_POINT_DISTANCE_ATTENUATION_ARB, attenuation_eqn);
		glPointSize( 1 );
	}
	else if (size_units == PIXELS) {
		// Restore to default (aka, disable attenuation)
		/*
		if (scene.glext.ARB_point_parameters) {
			float attenuation_eqn[] = {1.0f, 0.0f, 0.0f};
			scene.glext.glPointParameterfvARB( GL_POINT_DISTANCE_ATTENUATION_ARB, attenuation_eqn);
		}
		*/
		if (points_shape == ROUND) {
			glPointSize( size );
		}
		else {
			glPointSize( size );
		}
	}

	// Finish GL state prep
	gl_disable ltg( GL_LIGHTING);
	gl_enable_client v( GL_VERTEX_ARRAY);
	gl_enable_client c( GL_COLOR_ARRAY);

	// Render opaque points (if any)
	if (opaque_points.size()) {
		const std::ptrdiff_t chunk = 256;
		opaque_iterator begin = opaque_points.begin();
		opaque_iterator end = opaque_points.end();
		while (begin < end) {
			std::ptrdiff_t block = std::min( chunk, end - begin);
			glColorPointer( 3, GL_FLOAT, sizeof(point_coord), &begin->color.red);
			glVertexPointer( 3, GL_DOUBLE, sizeof(point_coord), &begin->center.x);
			glDrawArrays( GL_POINTS, 0, block);
			begin += block;
		}
	}

	/*
	// Render translucent points (if any)
	if (!translucent_points.empty()) {
		const std::ptrdiff_t chunk = 256;
		translucent_iterator begin = translucent_points.begin();
		translucent_iterator end = translucent_points.end();
		while (begin < end) {
			std::ptrdiff_t block = std::min( chunk, end - begin);
			glColorPointer( 3, GL_FLOAT, sizeof(point_coord), &begin->color.red);
			glVertexPointer( 3, GL_DOUBLE, sizeof(point_coord), &begin->center.x);
			glDrawArrays( GL_POINTS, 0, block);
			begin += block;
		}
	}
	*/

	if (points_shape == ROUND) {
		glDisable( GL_POINT_SMOOTH);
	}
	check_gl_error();
}

vector
points::get_center() const
{
	if (degenerate() || points_shape != ROUND)
		return vector();
	vector ret;
	const double* pos_i = pos.data();
	for(size_t i=0; i<count; i++, pos_i+=3)
		ret += vector(pos_i);
	ret /= count;
	return ret;
}

void
points::gl_pick_render(view& scene)
{
	gl_render( scene);
}

void
points::grow_extent( extent& world)
{
	if (degenerate())
		return;
	const double* pos_i = pos.data();
	const double* pos_end = pos.end();
	if (size_units == PIXELS)
		for ( ; pos_i < pos_end; pos_i += 3)
			world.add_point( vector(pos_i));
	else
		for ( ; pos_i < pos_end; pos_i += 3)
			world.add_sphere( vector(pos_i), size);
	world.add_body();
}

void points::outer_render(view& v ) {
	gl_render(v);  //< no materials
}


} } // !namespace cvisual::python
