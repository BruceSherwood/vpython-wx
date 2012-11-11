#include "text.hpp"
#include "font_renderer.hpp"
#include "util/gl_enable.hpp"
#include "util/errors.hpp"
#include "boost/algorithm/string.hpp"
#include "text_adjust.hpp"

namespace cvisual {

using std::wstring;

typedef std::map< std::pair<wstring, int>, boost::shared_ptr<font> >
	fontcache_t;
fontcache_t font_cache;

font::font( font_renderer* fr ) : renderer(fr) {}

boost::shared_ptr<font>
font::find_font( const wstring& desc, int height ) {
	if (height <= 0) height = 13;

	std::vector<wstring> fonts;
	if (desc.size())
		boost::split( fonts, desc, boost::is_any_of(L",") );
		
	// normalize font names
	for(size_t i=0; i<fonts.size(); i++) {
		boost::trim(fonts[i]);
		boost::to_lower(fonts[i]);
		if (fonts[i] == L"sans") fonts[i] = L"sans-serif";
	}

	fonts.push_back( L"sans-serif" );	//< default and last resort

	{
		// Attempt to use consistent fonts for generic font families across platforms
		std::vector<wstring> real_fonts;
		for(size_t i=0; i<fonts.size(); i++) {
			if		(fonts[i] == L"sans-serif")	real_fonts.push_back( L"Verdana" );
			else if (fonts[i] == L"serif")		real_fonts.push_back( L"Times New Roman" );
			else if (fonts[i] == L"monospace")	real_fonts.push_back( L"Courier New" );
			real_fonts.push_back( fonts[i] );
		}
		fonts.swap( real_fonts );
	}
		
	for(size_t i=0; i<fonts.size(); i++) {
		boost::shared_ptr<font>& f = font_cache[ std::make_pair( fonts[i], int(height*text_adjust+0.5)) ];
		if (!f) {
			f.reset( new font( new font_renderer( fonts[i], int(height*text_adjust+0.5) ) ) );
			f->self = f;
		}
				
		if ( f->renderer->ok() )
			return f;
	}
	
	throw std::logic_error( "Platform font_renderer doesn't recognize required generic font names." );
}

boost::shared_ptr<layout> 
font::lay_out( const wstring& text ) {
	shared_ptr<font> me( self );
	return boost::shared_ptr<layout>(
		new layout( me, text ) );
}

layout::layout( const boost::shared_ptr<font>& font, const wstring& text )
 : tx( font, text )
{
}

vector layout::extent( const view& v ) {
	tx.gl_activate(v);  //< a little tacky
	return vector( tx.width, tx.height );
}

void layout::gl_render( const view& v, const vector& pos_ll ) {
	gl_enable enTex( tx.enable_type() );
	tx.gl_activate(v);

	glTranslated( pos_ll.x, pos_ll.y, pos_ll.z );

	if (tx.internal_format == GL_ALPHA) {
		// Simple antialiasing
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
		draw_quad();
	} else {
		// For color antialiasing, we want to render "spectral alpha", i.e.
		//   framebuffer = framebuffer * (1-texture) + color * texture
		// OpenGL doesn't support spectral alpha, so we do it in two passes:
		//   framebuffer = framebuffer * (1-texture)
		//   framebuffer = framebuffer + color * texture

		glBlendFunc( GL_ZERO, GL_ONE_MINUS_SRC_COLOR );
		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
		draw_quad();

		glBlendFunc( GL_ONE, GL_ONE );
		glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
		draw_quad();
	}
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	
	check_gl_error();
}

void layout::draw_quad() {
	glBegin(GL_QUADS);
	for(int i=0; i<4; i++) {
		glTexCoord2d( tx.tcoord[i].x, tx.tcoord[i].y );
		tx.coord[i].gl_render();
	}
	glEnd();

}

layout_texture::layout_texture( const boost::shared_ptr<font>& _font, const wstring& _text )
 : text_font(_font), text(_text)
{
	damage();
}

layout_texture::~layout_texture() {
	
}

void layout_texture::gl_init( const view& v ) {
	int type = enable_type();

	gl_enable tex( type );

	GLuint handle;
	glGenTextures(1, &handle);
	set_handle( v, handle );
	glBindTexture(type, handle);

	// No filtering - we want the exact pixels from the texture
	glTexParameteri( type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri( type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Calls this->set_image()
	text_font->renderer->gl_render_to_texture( v, text, *this );
}

void layout_texture::set_image( int width, int height, int gl_internal_format, int gl_format, 
								int gl_type, int alignment, void* data )
{
	int bottom_up = height < 0;
	if (height < 0) height = -height;

	int type = enable_type();
	int tx_width, tx_height;
	double tc_x, tc_y;
	
	if ( type == GL_TEXTURE_2D ) {
		tx_width = next_power_of_two( width );
		tx_height = next_power_of_two( height );
		tc_x = (double)width / tx_width;
		tc_y = ((double)height) / (tx_height);
	} else {
		assert( false );	// TODO: Rectangular texture support
	}

	glPixelStorei( GL_UNPACK_ALIGNMENT, alignment );
	glPixelStorei( GL_UNPACK_ROW_LENGTH, width );
	
	check_gl_error();
	
	/*{
	std::ostringstream os;
	os << "glTexImage2D( " << type << ", 0, " << gl_internal_format << ", " 
		<< tx_width << ", " << tx_height << ", 0, " << gl_format << ", "
		<< gl_type << ", NULL );\n";
	write_stderr( os.str() );
	}*/
	
	glTexImage2D( type, 0, gl_internal_format, tx_width, tx_height, 0, gl_format, gl_type, NULL );
	check_gl_error();
	glTexSubImage2D(type, 0, 0, 0, width, height, gl_format, gl_type, data);
	check_gl_error();

	glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
	glPixelStorei( GL_UNPACK_ROW_LENGTH, 0 );

	this->width = width;
	this->height = height;
	this->internal_format = gl_internal_format;

	coord[0] = vector();
	coord[1] = vector(0, -height);
	coord[2] = vector(width, -height);
	coord[3] = vector(width, 0);
	
	tcoord[0^bottom_up] = vector();
	tcoord[1^bottom_up] = vector(0, tc_y);
	tcoord[2^bottom_up] = vector(tc_x, tc_y);
	tcoord[3^bottom_up] = vector(tc_x, 0);
}

}  // namespace cvisual
