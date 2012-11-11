#ifndef VPYTHON_GTK2_FONT_RENDERER_HPP
#define VPYTHON_GTK2_FONT_RENDERER_HPP

// See text.hpp for public interface

#include "text.hpp"
#include <glibmm/refptr.h>
#include <glibmm/ustring.h>
#include <pangomm.h>
#include <pango/pangoft2.h>

namespace cvisual {

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

 private:
	Glib::RefPtr<Pango::Context> ft2_context;
};

extern Glib::ustring w2u( const std::wstring& );

} // namespace cvisual

#endif
