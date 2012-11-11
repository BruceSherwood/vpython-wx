#ifndef VPYTHON_TEXT_HPP
#define VPYTHON_TEXT_HPP
#pragma once

/* New text rendering organization:

	This module defines the platform-independent part of the text renderer.
	
	The public classes are font and layout.  layout_texture is only for the use
	of the platform-specific font_renderer.
	
	On all platforms, text rendering is expected to work in essentially the same 
	way: an entire text string is rendered by the platform's text renderer into a
	texture (layout), which is then rendered as many times as called for.
	
	Each platform needs to provide {platform}/font_renderer.hpp with the following
	public interface (in namespace cvisual):

	class font_renderer {
	 public:

		// Create a font_renderer for the requested font.
		// Must support 'verdana' or 'sans-serif'
		// Should support 'times new roman' or 'serif', and 'courier new' or 'monospace'
		font_renderer( const std::wstring& description, int height );
		
		// Returns true if the requested font was available.
		bool ok();
		
		// Render text and call tx.set_image()
		void gl_render_to_texture( const struct view&, const std::wstring& text, layout_texture& tx );
	};
*/

#include "util/texture.hpp"
#include "util/vector.hpp"

namespace cvisual {

class font;
class layout;

class layout_texture : texture {
 public: // But only for use by font_renderer!
 
	// Takes similar parameters to glTexImage2D, but always accepts rectangular textures.
	// Pass a negative height if the image is bottom-up.
	// alignment is GL_UNPACK_ALIGNMENT
	// Typically the format should be either GL_ALPHA (for simple antialiasing) or 
	//   GL_RGB or GL_BGR_EXT (for color antialiasing e.g. ClearType)
	void set_image( int width, int height, int gl_internal_format, int gl_format, int gl_type, int alignment, void* data );

 private:
	layout_texture( const boost::shared_ptr<font>& text_font, const std::wstring& text );
	~layout_texture();
 
	boost::shared_ptr<font> text_font;
	std::wstring text;

	virtual void gl_init( const view& );
	
	friend class layout;
 	vector coord[4];
 	vector tcoord[4];
 	int width, height;
 	int internal_format;
};

class font {
 public:
	// Call this to get a font.  If possible, call only when the font changes.
	static boost::shared_ptr<font> 
	find_font( const std::wstring& desc = std::wstring(), int height = -1);

	// Get a layout for some text.  This needn't be called with an OpenGL
	// context, but it should be called as infrequently as possible (only when
	// text changes).
	boost::shared_ptr<layout> 
	lay_out( const std::wstring& text );

 private:
	friend class layout_texture;
	font( class font_renderer* );
	boost::weak_ptr<font> self;
	boost::scoped_ptr< class font_renderer > renderer;
};

class layout {
 public:
	// Renders the text with its lower left hand corner (NOT its baseline)
	// at the given position.
	void gl_render( const view& v, const vector& pos_ll );
	
	// Return the size of the text in pixels (x,y,0)
	vector extent( const view& );

 private:
	friend class font;

	layout( const boost::shared_ptr<font>& font, const std::wstring& text );
	void draw_quad();

	layout_texture tx;
};

} // namespace cvisual

#endif
