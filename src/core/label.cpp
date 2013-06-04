// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "label.hpp"
#include "util/errors.hpp"
#include <numpy/arrayobject.h> // Required to reference PyArrayObject

#include <sstream>
#include <iostream>

#include <boost/scoped_array.hpp>
using boost::scoped_array;

namespace cvisual {
using namespace boost::python::numeric;

using boost::python::import;
using boost::python::object;

bool setup_py_get_bitmap = true;
boost::python::object py_get_bitmap;

label::label()
	: pos(0, 0, 0),
	space(0),
	xoffset(0),
	yoffset(0),
	border(5),
	font_description(),
	font_size(13),
	text_changed(false),
	box_enabled(true),
	line_enabled(true),
	linecolor( color),
	opacity(0.66f),
	handle(0)
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
	text_changed(false),
	box_enabled( other.box_enabled),
	line_enabled( other.line_enabled),
	linecolor( other.linecolor),
	opacity( other.opacity),
	handle(0)
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
	text_changed = true;
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
	text_changed = true;
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
	text_changed = true;
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
	text_changed = true;
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

vector
label::get_center() const
{
	return pos;
}

void label::grow_extent( extent& e)
{
	e.add_point( pos );
}

void
label::set_handle( const view&, unsigned h ) {
	if (handle) on_gl_free.free( boost::bind( &gl_free, handle ) );

	handle = h;
	on_gl_free.connect( boost::bind(&label::gl_free, handle) );
}

void
label::gl_free(GLuint handle)
{
	glDeleteTextures(1, &handle);
}

void
label::set_primitive_object( boost::python::object obj)
{
	primitive_object = obj;
}

boost::python::object
label::get_primitive_object()
{
	return primitive_object;
}

void
label::get_bitmap()
// Call get_bitmap function in visual_common/primitives.py, which
// in turn calls set_bitmap below to set up bitmap RGBA unsigned bytes
// and bitmap_width and bitmap_height.
{
	if (setup_py_get_bitmap) {
		py_get_bitmap = import("visual_common.primitives").attr("get_bitmap");
		setup_py_get_bitmap = false;
	}
	py_get_bitmap(primitive_object);
}

void
label::set_bitmap(array bm, int width, int height, int back0, int back1, int back2) {
	// set_bitmap is called from primitives.py/get_bitmap
	// bm.data is RGB unsigned bytes
	// http://mail.python.org/pipermail/cplusplus-sig/2003-March/003202.html :
	unsigned char* data = (unsigned char*)((PyArrayObject*) bm.ptr())->data;
	bitmap_width = width;
	bitmap_height = height;
	text_changed = true;
	bitmap = new unsigned char[4*width*height];

	unsigned char b;
	for(int j=0; j<height; j++) {
		for(int i=0; i<width; i++) {
			bool is_background = true;

			b = data[3*width*j + 3*i];
			bitmap[4*width*j + 4*i] = b;
			if (b != back0) is_background = false;

			b = data[3*width*j + 3*i + 1];
			bitmap[4*width*j + 4*i + 1] = b;
			if (b != back1) is_background = false;

			b = data[3*width*j + 3*i + 2];
			bitmap[4*width*j + 4*i + 2] = b;
			if (b != back2) is_background = false;

			bitmap[4*width*j + 4*i + 3] =  (is_background) ? 0 : 255;
		}
	}

	/*
	// The following output demonstrates that bitmap is in fact
	// an array of the correct bytes to make the correct image.
	std::cout << "-------------------------------------" << std::endl;
	std::cout << bitmap_width << ", " << bitmap_height << std:: endl;
	for (int j=0; j<bitmap_height; j++) {
	    std::cout << std::endl;
	    for (int i=0; i<bitmap_width; i++) {
	        std::cout << "(";
	        for (int n=0; n<4; n++) {
	            std::cout << (int)bitmap[4*bitmap_width*j + 4*i + n] << ", ";
	        }
	        std::cout << "), ";
	    }
	}
	*/
}

// Using alpha-channel bitmap:
// http://stackoverflow.com/questions/5762622/opengl-textures-replacing-black-white-with-color

// Texture details:
// http://www.glprogramming.com/red/chapter09.html

void
label::gl_initialize( const view& v ) {

	int bottom_up = bitmap_height < 0;
	if (bitmap_height < 0) bitmap_height = -bitmap_height;

	int tx_width, tx_height;
	double tc_x, tc_y;

	// next_power_of_two is in texture.cpp
	tx_width = (int)next_power_of_two( bitmap_width );
	tx_height = (int)next_power_of_two( bitmap_height );
	tc_x = ((double)bitmap_width) / tx_width;
	tc_y = ((double)bitmap_height) / tx_height;

	const int type = GL_TEXTURE_2D;

	gl_enable tex( type );

	if (!handle) {
		glGenTextures(1, &handle);
		set_handle( v, handle );
	}
	glBindTexture(type, handle);

	// No filtering - we want the exact pixels from the texture
	glTexParameteri( type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri( type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
	glPixelStorei( GL_UNPACK_ROW_LENGTH, bitmap_width );

	check_gl_error();

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tx_width, tx_height, 0,
					GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	check_gl_error();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bitmap_width, bitmap_height,
					GL_RGBA, GL_UNSIGNED_BYTE, bitmap);
	check_gl_error();

	glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
	glPixelStorei( GL_UNPACK_ROW_LENGTH, 0 );

	coord[0] = vector();
	coord[1] = vector(0, -bitmap_height);
	coord[2] = vector(bitmap_width, -bitmap_height);
	coord[3] = vector(bitmap_width, 0);

	tcoord[0^bottom_up] = vector();
	tcoord[1^bottom_up] = vector(0, tc_y);
	tcoord[2^bottom_up] = vector(tc_x, tc_y);
	tcoord[3^bottom_up] = vector(tc_x, 0);
}

void label::gl_render_to_quad( const view& v, const vector& text_pos ) {
	gl_initialize(v);

	gl_enable tex1( GL_TEXTURE_2D );
	glBindTexture(GL_TEXTURE_2D, handle);

	glTranslated( text_pos.x, text_pos.y, text_pos.z );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	draw_quad();

	gl_disable tex2(GL_TEXTURE_2D);
	check_gl_error();
		}

void
label::draw_quad() {
	glBegin(GL_QUADS);
	for(int i=0; i<4; i++) {
		glTexCoord2d( tcoord[i].x, tcoord[i].y );
		coord[i].gl_render();
	}
	glEnd();
}

void
label::gl_render(view& scene)
{
	if (text_changed) {
		// Call get_bitmap in primitives.py, which calls set_bitmap in this file
		// to set bitmap, bitmap_width, and bitmap_height
		get_bitmap();
		text_changed = false;
	}

	// Compute the width of the text box.
	double box_width = bitmap_width + 2*border;

	// Compute the positions of the text in the text box, and the height of the
	// text box.  The text positions are relative to the lower left corner of
	// the text box.
	double box_height = bitmap_height + 2*border;

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
		gl_render_to_quad(scene, text_pos);

	} glMatrixMode( GL_MODELVIEW); } // Pops the matrices back off the stack
	list.gl_compile_end();
	check_gl_error();
	scene.screen_objects.insert( std::make_pair(pos, list));
}

} // !namespace cvisual
