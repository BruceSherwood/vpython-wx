// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "frame.hpp"
#include "util/errors.hpp"
#include <iostream>

#include <algorithm>

namespace cvisual {

frame::frame()
	: pos( 0, 0, 0),
	axis( 1, 0, 0),
	up( 0, 1, 0)
	// Disable frame.scale in Visual 4.0
	//scale( 1.0, 1.0, 1.0)
{
}

frame::frame( const frame& other)
	: renderable( other),
	pos(other.pos.x, other.pos.y, other.pos.z),
	axis(other.axis.x, other.axis.y, other.axis.z),
	up(other.up.x, other.up.y, other.up.z)
	// scale(other.scale.x, other.scale.y, other.scale.z)
{
}

frame::~frame()
{
}

void
frame::set_pos( const vector& n_pos)
{
	pos = n_pos;
}

shared_vector&
frame::get_pos()
{
	return pos;
}

void
frame::set_x( double x)
{
	pos.set_x( x);
}

double
frame::get_x()
{
	return pos.x;
}

void
frame::set_y( double y)
{
	pos.set_y( y);
}

double
frame::get_y()
{
	return pos.y;
}

void
frame::set_z( double z)
{
	pos.set_z( z);
}

double
frame::get_z()
{
	return pos.z;
}

void
frame::set_axis( const vector& n_axis)
{
	vector a = axis.cross(n_axis);
	if (a.mag() == 0.0) {
		axis = n_axis;
	} else {
		double angle = n_axis.diff_angle(axis);
		axis = n_axis.mag()*axis.norm();
		rotate(angle, a, pos);
	}
}

shared_vector&
frame::get_axis()
{
	return axis;
}

void
frame::set_up( const vector& n_up)
{
	up = n_up;
}

shared_vector&
frame::get_up()
{
	return up;
}

/*
void
frame::set_scale( const vector& n_scale)
{
	scale = n_scale;
}

shared_vector&
frame::get_scale()
{
	return scale;
}
*/

void
frame::rotate( double angle, const vector& _axis, const vector& origin)
{
	tmatrix R = rotation( angle, _axis, origin);
	vector fake_up = up;
	if (!axis.cross( fake_up)) {
		fake_up = vector( 1,0,0);
		if (!axis.cross( fake_up))
			fake_up = vector( 0,1,0);
	}
    {
        pos = R * pos;
        axis = R.times_v( axis);
        up = R.times_v( fake_up);
    }
}

vector
frame::world_zaxis() const
{
	vector z_axis;
	if (std::fabs(axis.dot(up) / std::sqrt( up.mag2() * axis.mag2())) > 0.98) {
		if (std::fabs(axis.norm().dot( vector(-1,0,0))) > 0.98)
			z_axis = axis.cross( vector(0,0,1)).norm();
		else
			z_axis = axis.cross( vector(-1,0,0)).norm();
	}
	else {
		z_axis = axis.cross( up).norm();
	}
	return z_axis;
}

vector
frame::frame_to_world( const vector& p) const
{
	vector z_axis = world_zaxis();
	vector y_axis = z_axis.cross(axis).norm();
	vector x_axis = axis.norm();
	vector inworld = pos + p.x*x_axis + p.y*y_axis + p.z*z_axis;

	return inworld;
}

vector
frame::world_to_frame( const vector& p) const
{
	vector z_axis = world_zaxis();
	vector y_axis = z_axis.cross(axis).norm();
	vector x_axis = axis.norm();
	vector v = p - pos;
	vector inframe = vector(v.dot(x_axis), v.dot(y_axis), v.dot(z_axis));

	return inframe;
}

tmatrix
frame::frame_world_transform( const double gcf) const
{
	// Performs a reorientation transform.
	// ret = translation o reorientation
	tmatrix ret;

	vector z_axis = world_zaxis();
	vector y_axis = z_axis.cross(axis).norm();
	vector x_axis = axis.norm();

	ret.x_column( x_axis);
	ret.y_column( y_axis);
	ret.z_column( z_axis);

	ret.w_column( pos * gcf);
	ret.w_row();
	return ret;
}

tmatrix
frame::world_frame_transform() const
{
	// Performs a reorientation transform.
	// ret = translation o reorientation
	// ret = ireorientation o itranslation.
	// Robert Xiao pointed out that this was incorrect, and he proposed
	// replacing it with inverse(ret, frame_world_transform(1.0)).
	// However, comparison with Visual 3 showed that there were
	// simply minor errors to be fixed.
	tmatrix ret;

	vector z_axis = world_zaxis();
	vector y_axis = z_axis.cross(axis).norm();
	vector x_axis = axis.norm();

	ret(0,0) = x_axis.x;
	ret(0,1) = x_axis.y;
	ret(0,2) = x_axis.z;
	ret(0,3) = -(pos * x_axis).sum();
	ret(1,0) = y_axis.x;
	ret(1,1) = y_axis.y;
	ret(1,2) = y_axis.z;
	ret(1,3) = -(pos * y_axis).sum();
	ret(2,0) = z_axis.x;
	ret(2,1) = z_axis.y;
	ret(2,2) = z_axis.z;
	ret(2,3) = -(pos * z_axis).sum();

	ret.w_row();

	return ret;
}

void
frame::add_renderable( shared_ptr<renderable> obj)
{
	// Driven from visual/primitives.py set_visible
	if (!obj->translucent())
		children.push_back( obj);
	else
		trans_children.push_back( obj);
}

void
frame::remove_renderable( shared_ptr<renderable> obj)
{
	// Driven from visual/primitives.py set_visible
	if (!obj->translucent()) {
		std::remove( children.begin(), children.end(), obj);
		children.pop_back();
	}
	else {
		std::remove( trans_children.begin(), trans_children.end(), obj);
		trans_children.pop_back();
	}
}

std::vector<shared_ptr<renderable> >
frame::get_objects()
{
	std::vector<shared_ptr<renderable> > ret;
	get_children(ret);
	return ret;
}

shared_ptr<renderable>
frame::lookup_name(
	const unsigned int* name_top,
	const unsigned int* name_end)
{
	assert( name_top < name_end);
	assert( *name_top < children.size() + trans_children.size());
	using boost::dynamic_pointer_cast;

	shared_ptr<renderable> ret;
	unsigned int size = 0;
	const_child_iterator i( children.begin());
	const_child_iterator i_end( children.end());
	while (i != i_end) {
		if (*name_top == size) {
			ret = *i.base();
			break;
		}
		size++;
		++i;
	}
	if (!ret)
		ret = trans_children[*(name_top) - size];

	if (name_end - name_top > 1) {
		frame* ref_frame = dynamic_cast<frame*>(ret.get());
		assert( ref_frame != NULL);
		return ref_frame->lookup_name(name_top + 1, name_end);
	}
	else
		return ret;
}

vector
frame::get_center() const
{
	return pos;
}

void
frame::gl_render(view& v)
{
	view local(v); local.apply_frame_transform(world_frame_transform());
    tmatrix fwt = frame_world_transform(v.gcf);
	{
		gl_matrix_stackguard guard( fwt);

		child_iterator i(children.begin());
		child_iterator i_end(children.end());
		while (i != i_end) {
			if (i->translucent()) {
				// See display_kernel::draw().
				trans_children.push_back( *i.base());
				i = children.erase(i.base());
				continue;
			}
			i->outer_render(local);
			i++;
		}

		// Perform a depth sort of the transparent children from forward to backward.
		if (!trans_children.empty()) {
			opacity = 0.5;  //< TODO: BAD HACK
		}
		if (trans_children.size() > 1)
			std::stable_sort( trans_children.begin(), trans_children.end(),
				z_comparator( (pos*v.gcf - v.camera).norm()));

		for (trans_child_iterator i = trans_children.begin();
			i != trans_child_iterator(trans_children.end());
			++i)
		{
			i->outer_render(local);
		}
	}
	typedef std::multimap<vector, displaylist, z_comparator>::iterator screen_iterator;
	screen_iterator i( local.screen_objects.begin());
	screen_iterator i_end( local.screen_objects.end());
  //  v.screen_objects.clear();
	while (i != i_end) {
		v.screen_objects.insert( std::make_pair( fwt*i->first, i->second));
		++i;
	}
	//check_gl_error();
}

void
frame::gl_pick_render(view& scene)
{
	// TODO: This needs to construct a valid local view!
	// Push name
	glPushName(0);
	{
		gl_matrix_stackguard guard( frame_world_transform(scene.gcf));
		//gl_matrix_stackguard guard( frame_world_transform(1.0));
		child_iterator i( children.begin());
		child_iterator i_end( children.end());
		// The unique integer to pass to OpenGL.
		unsigned int name = 0;
		while (i != i_end) {
			glLoadName(name);
			i->gl_pick_render( scene);
			++i;
			++name;
		}

		trans_child_iterator j( trans_children.begin());
		trans_child_iterator j_end( trans_children.end());
		while (j != j_end) {
			glLoadName(name);
			j->gl_pick_render(scene);
			++j;
			++name;
		}
	}
	// Pop name
	glPopName();
	//check_gl_error();
}

void
frame::grow_extent( extent& world)
{
	extent local( world, frame_world_transform(1.0) );
	child_iterator i( children.begin());
	child_iterator i_end( children.end());
	for (; i != i_end; ++i) {
		i->grow_extent( local);
		local.add_body();
	}
	trans_child_iterator j( trans_children.begin());
	trans_child_iterator j_end( trans_children.end());
	for ( ; j != j_end; ++j) {
		j->grow_extent( local);
		local.add_body();
	}
}

void frame::render_lights( view& world ) {
	// TODO: this is expensive, especially if there are no lights at all in the frame!
	view local( world ); local.apply_frame_transform(world_frame_transform());

 	child_iterator i( children.begin());
	child_iterator i_end( children.end());
	for (; i != i_end; ++i)
		i->render_lights( local );
	trans_child_iterator j( trans_children.begin());
	trans_child_iterator j_end( trans_children.end());
	for ( ; j != j_end; ++j)
		j->render_lights( local );

	// Transform lights back into scene
	if ( world.light_count[0] != local.light_count[0] ) {
		tmatrix fwt = frame_world_transform(world.gcf);
		world.light_pos.resize( local.light_pos.size() );
		world.light_color.resize( local.light_color.size() );
		for(int l = world.light_count[0]; l < local.light_count[0]; l++) {
			int li = l*4;
			vertex v( local.light_pos[li], local.light_pos[li+1], local.light_pos[li+2], local.light_pos[li+3] );
			v = fwt * v;
			for(int d=0; d<4; d++) {
				world.light_pos[li+d] = v[d];
				world.light_color[li+d] = local.light_color[li+d];
			}
		}
		world.light_count[0] = local.light_count[0];
	}
}

void frame::get_children( std::vector< boost::shared_ptr<renderable> >& all )
{
	all.insert( all.end(), children.begin(), children.end() );
	all.insert( all.end(), trans_children.begin(), trans_children.end() );
}

void frame::outer_render(cvisual::view& v) {
  gl_render(v);
}

} // !namespace cvisual
