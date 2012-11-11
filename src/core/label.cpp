// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "label.hpp"
#include "util/errors.hpp"
#include "util/gl_enable.hpp"

#include <sstream>
#include <iostream>

#include <boost/scoped_array.hpp>
using boost::scoped_array;

namespace cvisual {

label::label()
	: pos(0, 0, 0),
	space(0),
	xoffset(0),
	yoffset(0),
	border(5),
	font_description(), // equivalent to label.font="sans"
	font_size(-1), // equivalent to label.height=13 (see text.cpp)
	box_enabled(true),
	line_enabled(true),
	linecolor( color),
	opacity(0.66f),
	text_changed(true)
{
	background = rgb(0., 0., 0.);
}

label::label( const label& other)
	: renderable( other),
	pos( other.pos.x, other.pos.y, other.pos.z),
	space( other.space),
	xoffset( other.xoffset),
	yoffset( other.yoffset),
	border( other.border),
	font_description( other.font_description),
	font_size( other.font_size),
	box_enabled( other.box_enabled),
	line_enabled( other.line_enabled),
	linecolor( other.linecolor),
	opacity( other.opacity),
	text( other.text),
	text_changed( true)
{
	background = rgb(0., 0., 0.);
}

label::~label()
{
}

void
label::set_pos( const vector& n_pos)
{
	pos = n_pos;
}

shared_vector&
label::get_pos()
{
	return pos;
}

void
label::set_x( double x)
{
	pos.set_x( x);
}

double
label::get_x()
{
	return pos.x;
}

void
label::set_y( double y)
{
	pos.set_y( y);
}

double
label::get_y()
{
	return pos.y;
}

void
label::set_z( double z)
{
	pos.set_z( z);
}

double
label::get_z()
{
	return pos.z;
}

void
label::set_color( const rgb& n_color)
{
	color = n_color;
}

rgb
label::get_color()
{
	return color;
}

void
label::set_red( float r)
{
	color.red = r;
}

double
label::get_red()
{
	return color.red;
}

void
label::set_green( float g)
{
	color.green = g;
}

double
label::get_green()
{
	return color.green;
}

void
label::set_blue( float b)
{
	color.blue = b;
}

double
label::get_blue()
{
	return color.blue;
}

double
label::get_opacity()
{
	return opacity;
}

void
label::set_opacity( float o)
{
	opacity = o;
}

void
label::set_text( const std::wstring& t )
{
	text = t;
	text_changed = true;
}


std::wstring
label::get_text()
{
	return text;
}

void
label::set_space( double n_space)
{
	space = n_space;
}

double
label::get_space()
{
	return space;
}

void
label::set_xoffset( double n_xoffset)
{
	xoffset = n_xoffset;
}

double
label::get_xoffset()
{
	return xoffset;
}

void
label::set_yoffset( double n_yoffset)
{
	yoffset = n_yoffset;
}

double
label::get_yoffset()
{
	return yoffset;
}

void
label::set_border( double n_border)
{
	border = n_border;
}

double
label::get_border()
{
	return border;
}

void
label::set_font_family( const std::wstring& name)
{
	font_description = name;
	text_changed = true;
}

std::wstring
label::get_font_family()
{
	return font_description;
}

void
label::set_font_size( double n_size)
{
	font_size = n_size;
	text_changed = true;
}

double
label::get_font_size()
{
	return font_size;
}

void
label::render_box( bool enable)
{
	box_enabled = enable;
}

bool
label::has_box()
{
	return box_enabled;
}

void
label::render_line( bool enable)
{
	line_enabled = enable;
}

bool
label::has_line()
{
	return line_enabled;
}

void
label::set_linecolor( const rgb& n_color)
{
	linecolor = n_color;
}

rgb
label::get_linecolor()
{
	return linecolor;
}

void
label::set_background( const rgb& n_background)
{
	background = n_background;
}

rgb
label::get_background()
{
	return background;
}

void label::grow_extent( extent& e)
{
	e.add_point( pos );
}

void
label::gl_render(view& scene)
{
	if (text_changed) {
		boost::shared_ptr<font> texmap_font =
			font::find_font( font_description, int(font_size));
		if (text.empty())
			text_layout = texmap_font->lay_out( L" " );
		else
			text_layout = texmap_font->lay_out( text);
		text_changed = false;
	}
	// Compute the width of the text box.
	vector extents = text_layout->extent( scene );
	double box_width = extents.x + 2.0*border;

	// Compute the positions of the text in the text box, and the height of the
	// text box.  The text positions are relative to the lower left corner of
	// the text box.
	double box_height = extents.y + 2.0*border;

	vector text_pos( border, box_height - border);

	clear_gl_error();
	vector label_pos = pos.scale(scene.gcfvec);
	tmatrix lst = tmatrix().gl_projection_get() * tmatrix().gl_modelview_get();
	{
		tmatrix translate;
		translate.w_column( label_pos);
		lst = lst * translate;
	}
	vector origin = (lst * vertex(vector(), 1.0)).project();

	// It is very important to make sure that the texture is positioned
	// accurately at a screen pixel location, to avoid artifacts around the texture.
	double kx = scene.view_width/2.0;
	double ky = scene.view_height/2.0;
	if (origin.x >= 0) {
		origin.x = ((int)(kx*origin.x+0.5))/kx;
	} else {
		origin.x = -((int)(-kx*origin.x+0.5))/kx;
	}
	if (origin.y >= 0) {
		origin.y = ((int)(ky*origin.y+0.5))/ky;
	} else {
		origin.y = -((int)(-ky*origin.y+0.5))/ky;
	}
	double halfwidth = (int)(0.5*box_width+0.5);
	double halfheight = (int)(0.5*box_height+0.5);

	rgb stereo_linecolor = linecolor;
	if (scene.anaglyph)
		if (scene.coloranaglyph)
			stereo_linecolor = linecolor.desaturate();
		else
			stereo_linecolor = linecolor.grayscale();

	displaylist list;
	list.gl_compile_begin();
	{
		stereo_linecolor.gl_set(1.0f);
		// Zero out the existing matrices, rendering will be in screen coords.
		gl_matrix_stackguard guard;
		tmatrix identity;
		identity.gl_load();
		glMatrixMode( GL_PROJECTION); { //< Zero out the projection matrix, too
		gl_matrix_stackguard guard2;
		identity.gl_load();

		glTranslated( origin.x, origin.y, origin.z);
		glScaled( 1.0/kx, 1.0/ky, 1.0);
		// At this point, all further translations are in direction of label space.
		if (space && (xoffset || yoffset)) {
			// Move the origin away from the body.
			vector space_offset = vector(xoffset, yoffset).norm() * std::fabs(space);
			glTranslated( space_offset.x, space_offset.y, space_offset.z);
		}
		// Optionally draw the line, and move the origin to the bottom left
		// corner of the text box.
		if (xoffset || yoffset) {
			if (line_enabled) {
				glBegin( GL_LINES);
					vector().gl_render();
					vector(xoffset, yoffset).gl_render();
				glEnd();
			}
			if (std::fabs(xoffset) > std::fabs(yoffset)) {
				glTranslated(
					xoffset + ((xoffset > 0) ? 0 : -2.0*halfwidth),
					yoffset - halfheight,
					0);
			}
			else {
				glTranslated(
					xoffset - halfwidth,
					yoffset + ((yoffset > 0) ? 0 : -2.0*halfheight),
					0);
			}
		}
		else {
			glTranslated( -halfwidth, -halfheight, 0.0);
		}

		if (opacity) {
			// Occlude objects behind the label.
			rgba( background[0], background[1], background[2], opacity).gl_set();
			glBegin( GL_QUADS);
				vector().gl_render();
				vector( 2.0*halfwidth, 0).gl_render();
				vector( 2.0*halfwidth, 2.0*halfheight).gl_render();
				vector( 0, 2.0*halfheight).gl_render();
			glEnd();
		}
		if (box_enabled) {
			// Draw a box around the text.
			stereo_linecolor.gl_set(1.0f);
			glBegin( GL_LINE_LOOP);
				vector().gl_render();
				vector( 2.0*halfwidth, 0).gl_render();
				vector( 2.0*halfwidth, 2.0*halfheight).gl_render();
				vector( 0, 2.0*halfheight).gl_render();
			glEnd();
		}

		// Render the text itself.
		color.gl_set(1.0f);
		text_layout->gl_render(scene, text_pos);
	} glMatrixMode( GL_MODELVIEW); } // Pops the matrices back off the stack
	list.gl_compile_end();
	check_gl_error();
	scene.screen_objects.insert( std::make_pair(pos, list));
}

vector
label::get_center() const
{
	return pos;
}

} // !namespace cvisual
