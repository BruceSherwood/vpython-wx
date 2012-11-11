#ifndef VPYTHON_UTIL_EXTENT_HPP
#define VPYTHON_UTIL_EXTENT_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/vector.hpp"
#include "util/tmatrix.hpp"

namespace cvisual {

class extent_data {
private:
	friend class extent;

	double cot_hfov, invsin_hfov, sin_hfov, cos_hfov;

	vector mins, maxs;
	double camera_z;

	size_t buffer_depth; ///< The required depth of the selection buffer.

	bool is_empty() const;

public:
	extent_data( double tan_hfov );

	// The following functions represent the interface for render_surface objects.
	/** Returns the center position of the scene in world space. */
	vector get_center() const;

	/** Returns distance that are nearest and farthest toward center along forward. */
	void get_near_and_far( const vector& forward, double& nearest, double& farthest) const;

	/** Determines the range that the axes need to be to include the bounding
	 * box.
	 */
	vector get_range( vector center) const;

	double get_camera_z() const { return camera_z; }

	/** Returns the size for the select buffer when rendering in select mode,
	 * after having traversed the world. */
	size_t get_select_buffer_depth() { return buffer_depth; }
};

/** A helper class to determine the extent of the rendered universe in world
	space.
	*/
class extent
{
 private:
	extent_data& data;
	tmatrix l_cw;
	int frame_depth;
 public:
	extent( extent_data& data, const tmatrix& local_to_centered_world );
	extent( extent& parent, const tmatrix& local_to_parent );
	~extent(); //< Might be necessary to "flush" local cached results into parent

	// The following functions represent the interface for renderable objects.
	/** Extend the range to include this point.
 		@param point a point in world space coordinates.
 	*/
	void add_point( vector point);
	/** Extend the range to include this sphere.
 		@param center The center of the sphere.
 		@param radius The radius of the bounding sphere.
 	*/
	void add_sphere( vector center, double radius);
	/** Extend the range to include the region of local coordinate system
	    from min to max **/
	void add_box( const tmatrix& local_to_world, const vector& min, const vector& max );
	/** Extend the range to include this circle */
	void add_circle( const vector& center, const vector& normal, double radius );

	/** Report the number of bodies that this object represents.  This is used
	 *  for the calculation of the hit buffer size.
	 */
	void add_body();
	/** See implementation of frame::grow_extent()
	 */
};

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_EXTENT_HPP
