#ifndef VPYTHON_RENDERABLE_HPP
#define VPYTHON_RENDERABLE_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/rgba.hpp"
#include "util/extent.hpp"
#include "util/displaylist.hpp"
#include "util/texture.hpp"
#include "util/gl_extensions.hpp"
#include <boost/shared_ptr.hpp>

#include <map>

namespace cvisual {

using boost::shared_ptr;
class renderable;

const int N_LIGHT_TYPES = 1;

/** A depth sorting criterion for STL-compatable sorting algorithms.  This
   implementation only performs 4 adds, 6 multiplies, and one comparison.  It
   could be made faster if the virtual function get_center() was somehow made
   non-virtual, but that isn't possible right now since some bodies
   have such a different notion of the "center" of the body compared to the other
   objects.
 */
class z_comparator
{
 private:
	/** A unit vector along the visual depth axis.*/
	vector forward;
 public:
	/** Create a new comparator based on the specified depth direction.
	 * @param fore A unit vector along the sorting axis.
	 */
	z_comparator( const vector& fore)
		: forward( fore) {}

	/** Apply the sorting criteria.
		@return true if lhs is farther away than rhs.
	*/
	inline bool operator()(
		const shared_ptr<renderable> lhs,
		const shared_ptr<renderable> rhs) const;

	/** Apply the sorting criteria.  This version is faster than the shared_ptr
		version above, by an amount that varies from OS to OS.
		@return true if lhs is farther away than rhs.
	*/
	inline bool operator()( const renderable* lhs, const renderable* rhs) const;

	/** Apply the sorting criteria.  This version is used by view::scen_objects
		to sort them in depth-order as they are added to it.
		@return true if lhs is farther away than rhs.
	*/
	inline bool operator()( const vector& lhs, const vector& rhs) const;

};

/** This primarily serves as a means of communicating information down to the
	various primitives that may or may not need it from the render_surface.  Most
	of the members are simply references to the real values in the owning
	render_surface.
*/
struct view
{
	/// The position of the camera in world space.
	vector camera;
	/// The direction the camera is pointing - a unit vector.
	vector forward;
	/// The center of the scene in world space.
	vector center;
	/// The true up direction of the scene in world space.
	vector up;

	/// The width of the viewport in pixels.
	int view_width;
	/// The height of the viewport in pixels.
	int view_height;
	/// True if the forward vector changed since the last rending operation.
	bool forward_changed;
	/// The Global Scaling Factor
	double gcf;
	/// The vector version of the Global Scaling Factor, for scene.uniform=0
	vector gcfvec;
	/// True if gcf changed since the last render cycle.
	bool gcf_changed;
	/// The user adjustment to the level-of-detail.
	int lod_adjust;
	/// True in anaglyph stereo rendering modes.
	bool anaglyph;
	/// True in coloranaglyph stereo rendering modes.
	bool coloranaglyph;
	double tan_hfov_x; ///< The tangent of half the horzontal field of view.
	double tan_hfov_y; ///< The tangent of half the vertical field of view.

	displaylist box_model;
	displaylist sphere_model[6];
	displaylist cylinder_model[6];
	displaylist cone_model[6];
	displaylist pyramid_model;

	gl_extensions& glext;

	tmatrix camera_world;

	int light_count[N_LIGHT_TYPES];
	std::vector<float> light_pos, light_color; // in eye coordinates!

	typedef std::multimap<vector, displaylist, z_comparator> screen_objects_t;
	mutable screen_objects_t screen_objects;

	bool enable_shaders;

	view( vector n_forward, vector n_center, int n_width,
		int n_height, bool n_forward_changed,
		double n_gcf, vector n_gcfvec,
		bool n_gcf_changed,
		gl_extensions& glext);

    /** Called on a copy of a parent view to make this a view in a child
     *  frame.  pft is a transform from the parent to the frame coordinate
     *  space.
     */
	void apply_frame_transform( const tmatrix& pft);

	// Compute the apparent diameter, in pixels, of a circle that is parallel
	// to the screen, with a center at pos, and some radius.  If pos is behind
	// the camera, it will return negative.
	double pixel_coverage( const vector& pos, double radius) const;
};

/** Virtual base class for all renderable objects and composites.
 */
class renderable
{
public:
	/** The base color of this body.  Ignored by the variable-color composites
	 * (curve, faces, frame).
	 */
	rgb color;

	virtual ~renderable();

	/** Applies materials and other general features and calls gl_render().
	 * For now, also calls refresh_cache(), but that might be moved back in
	 * order to make that function compute center. */
	virtual void outer_render(view&);


	/** Called when rendering for mouse hit testing.  Since the result is not
	 *  visible, subclasses should not perform texture mapping or blending,
	 * and should use the lowest-quality level of detail that covers the
	 * geometry.
	 */
	virtual void gl_pick_render(view&);

	/** Report the total extent of the object. */
	virtual void grow_extent( extent&);

	/** Report the approximate center of the object.  This is used for depth
	 * sorting of the transparent models.  */
	virtual vector get_center() const = 0;

	virtual void set_material( shared_ptr<class material> m );
	virtual shared_ptr<class material> get_material();

	virtual void get_material_matrix( const view&, tmatrix& out ) {};  // object coordinates -> material coordinates

	virtual bool translucent();

	virtual void render_lights( view& ) {}

	virtual bool is_light() { return false; }

	virtual void get_children( std::vector< boost::shared_ptr<renderable> >& all ) {}

protected:
	renderable();

	// Fully opaque is 1.0, fully transparent is 0.0:
	float opacity;

	shared_ptr<class material> mat;

	/** True if the object should be rendered on the screen. */
	bool visible;

	/** Called by outer_render when drawing to the screen.  The default
	 * is to do nothing.
	 */
	virtual void gl_render(view&);
};

inline bool
z_comparator::operator()(
	const shared_ptr<renderable> lhs, const shared_ptr<renderable> rhs) const
{
	return forward.dot( lhs->get_center()) > forward.dot( rhs->get_center());
}

inline bool
z_comparator::operator()( const renderable* lhs, const renderable* rhs) const
{
	return forward.dot( lhs->get_center()) > forward.dot( rhs->get_center());
}

inline bool
z_comparator::operator()( const vector& lhs, const vector& rhs) const
{
	return forward.dot( lhs) > forward.dot( rhs);
}


/** A utility function that clamps a value to within a specified range.
 * @param lower The lower bound for the value.
 * @param value The value to be clamped.
 * @param upper The upper bound for the value.
 * @return value if it is between lower and upper, otherwise one of the bounds.
 */
template <typename T>
T clamp( T const& lower, T const& value, T const& upper)
{
	if (lower > value)
		return lower;
	if (upper < value)
		return upper;
	return value;
}

} // !namespace cvisual

#endif // !defined VPYTHON_RENDERABLE_HPP
