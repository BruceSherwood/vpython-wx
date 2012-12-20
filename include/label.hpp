#ifndef VPYTHON_LABEL_HPP
#define VPYTHON_LABEL_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "util/gl_enable.hpp"
#include "wrap_gl.hpp"

//#include <string>
#include "util/gl_free.hpp"

#include <boost/python.hpp>

namespace cvisual {
using namespace boost::python::numeric;

class label : public renderable
{
 public:
	label();
	label( const label& other);
	virtual ~label();

	void set_pos( const vector& n_pos);
	shared_vector& get_pos();

	void set_x( double x);
	double get_x();

	void set_y( double y);
	double get_y();

	void set_z( double z);
	double get_z();

	void set_color( const rgb& n_color);
	rgb get_color();

	void set_red( float x);
	double get_red();

	void set_green( float x);
	double get_green();

	void set_blue( float x);
	double get_blue();

	void set_opacity( float);
	double get_opacity();
	
	void set_text( const std::wstring& t);
	std::wstring get_text();

	void set_space( double space);
	double get_space();

	void set_xoffset( double xoffset);
	double get_xoffset();

	void set_yoffset( double yoffset);
	double get_yoffset();

	void set_border( double border);
	double get_border();

	void set_font_family( const std::wstring& name);
	std::wstring get_font_family();

	void set_font_size(double);
	double get_font_size();

	void render_box( bool);
	bool has_box();

	void render_line( bool);
	bool has_line();

	void set_linecolor( const rgb& color);
	rgb get_linecolor();

	void set_background( const rgb& color);
	rgb get_background();

	void set_primitive_object( boost::python::object x);
	boost::python::object get_primitive_object();

	//void set_bitmap(char* data, int width, int height);
	void set_bitmap(array bm, int width, int height, int back0, int back1, int back2);

 protected:
	GLuint handle;
	static void gl_free( GLuint handle );

	// Sets handle and registers it to be freed at shutdown
	void set_handle( const view&, unsigned int handle );
	unsigned get_handle() { return handle; }

	bool text_changed;

 	vector coord[4];
 	vector tcoord[4];

	// In world space:
	shared_vector pos;
	double space;

	// In pixels:
	double xoffset;   // offset from pos + space to the box
	double yoffset;
	double border;    // space between text and box

	/// A common name for the font.
	std::wstring font_description;
	/// The nominal size of the font, in pixels.
	double font_size;

	bool box_enabled; ///< True to draw a box around the text
	bool line_enabled; ///< True to draw a line to the text.

	// bitmap_font* font;
	rgb linecolor; ///< The color of the lines in the label. (color is for text)
	float opacity; ///< The opacity of the background for the text.

	rgb background; // by default, the color of scene.background

	std::wstring text;

	virtual void gl_render(view&);
	virtual vector get_center() const;
	virtual void grow_extent( extent& );

	void gl_initialize(const view&);
	void gl_render_to_quad(const view& v, const vector& text_pos);
	void draw_quad();

	boost::python::object primitive_object;

	void get_bitmap();

	//unsigned char* bitmap;
	unsigned char* bitmap;
	int bitmap_width;
	int bitmap_height;
};

} // !namespace cvisual

#endif // !defined VPYTHON_LABEL_HPP
