#ifndef VPYTHON_FRAME_HPP
#define VPYTHON_FRAME_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "util/tmatrix.hpp"

#include <boost/iterator/indirect_iterator.hpp>
#include <vector>
#include <list>

namespace cvisual {

using boost::indirect_iterator;

/*
Operations on frame objects include:
get_center() : Use the average of all its children.
update_z_sort() : Never called.  Always re-sort this body's translucent children
	in gl_render().
gl_render() : Calls gl_render() on all its children.
grow_extent() : Calls grow_extent() for each of its children, then transforms
	the vertexes of the bounding box and uses those as its bounds.
gl_pick_render() : PushName() on to the Name Stack, and renders its children.
	When looking up names later, the render_core calls lookup_name() with a
	vector<uint>, which the frame uses to recursively look through frames to
	find the right object.

oolie case: When the frame is scaled up to a superhuge universe and the
	child is very small, the frame_world_transform may overflow OpenGL.  The
	problem lies in the scale variable.

another oolie: A transparent object that intersects a frame containing other
	transparent object's will not be rendered in the right order.
*/

class frame : public renderable
{
 private:
	shared_vector pos;
	shared_vector axis;
	shared_vector up;
	//shared_vector scale; // Disable frame.scale in Visual 4.0
	/** Establishes the coordinate system into which this object's children
 		are rendered.
 		@param gcf: the global correction factor, propogated from gl_render().
 	*/
	vector world_zaxis() const;
	tmatrix frame_world_transform( const double gcf) const;
	tmatrix world_frame_transform() const;

	std::list<shared_ptr<renderable> > children;
	typedef indirect_iterator<std::list<shared_ptr<renderable> >::iterator>
		child_iterator;
	typedef indirect_iterator<std::list<shared_ptr<renderable> >::const_iterator>
		const_child_iterator;

	std::vector<shared_ptr<renderable> > trans_children;
	typedef indirect_iterator<std::vector<shared_ptr<renderable> >::iterator>
		trans_child_iterator;
	typedef indirect_iterator<std::vector<shared_ptr<renderable> >::const_iterator>
		const_trans_child_iterator;

 public:
	frame();
	frame( const frame& other);
	virtual ~frame();
    void rotate( double angle, const vector& axis, const vector& origin);

	void add_renderable( shared_ptr<renderable> child);
	void remove_renderable( shared_ptr<renderable> child);
	std::vector<shared_ptr<renderable> > get_objects();

	void set_pos( const vector& n_pos);
	shared_vector& get_pos();

	void set_x( double x);
	double get_x();

	void set_y( double y);
	double get_y();

	void set_z( double z);
	double get_z();

	void set_axis( const vector& n_axis);
	shared_vector& get_axis();

	void set_up( const vector& n_up);
	shared_vector& get_up();

	vector frame_to_world(const vector& p) const;
	vector world_to_frame(const vector& p) const;

	// void set_scale( const vector& n_scale);
	// shared_vector& get_scale();

	// Lookup the target that belongs to this name.
	shared_ptr<renderable> lookup_name(
		const unsigned int* name_top, const unsigned int* name_end);

	virtual void get_children( std::vector< boost::shared_ptr<renderable> >& all );

 protected:
	virtual vector get_center() const;
	virtual void outer_render(view&);
	virtual void gl_render(view&);
	virtual void gl_pick_render(view&);
	virtual void grow_extent( extent&);
	virtual void render_lights( view& );
};

} // !namespace cvisual

#endif // !defined VPYTHON_FRAME_HPP
