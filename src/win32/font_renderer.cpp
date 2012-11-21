#include "font_renderer.hpp"
#include "util/errors.hpp"

namespace cvisual {

using std::wstring;

class win32_exception : std::exception {
 public:
	win32_exception( const char* desc ) : std::exception(desc) {
		// todo: report GetLastError(), see win32_write_critical in windisplay.cpp
	}
};

bool isClearTypeEnabled() {
	UINT smoothType = 0;
	// On versions of Windows < XP, this call should fail
	if (SystemParametersInfo( 0x200a, 0, &smoothType, 0 )) {  // SPI_GETFONTSMOOTHINGTYPE
		if (smoothType == 2) // FE_FONTSMOOTHINGCLEARTYPE
			return true;
	}
	return false;
}

static int CALLBACK ef_callback(ENUMLOGFONTEXW *,NEWTEXTMETRICEXW *,DWORD,LPARAM lParam) {
	*(bool*)lParam = true;
	return 0;
}

font_renderer::font_renderer( const wstring& description, int height ) {
	font_handle = NULL;
	
	// TODO: support generic "sans-serif", "serif", "monospace" families using lfPitchAndFamily.
	// Doesn't matter much because Windows machines pretty much always have "verdana", 
	// "times new roman", and "courier new".

	// Respect the users' preferences as to whether ClearType should be enabled.
	isClearType = isClearTypeEnabled();
	int quality = DEFAULT_QUALITY;
	if (isClearType) quality = 5;
	
	HDC sic = CreateIC( "DISPLAY", NULL, NULL, NULL );
	
	LOGFONTW lf;
	memset(&lf, 0, sizeof(lf));
	lf.lfHeight = -height;
	lf.lfOutPrecision = OUT_TT_PRECIS;
	lf.lfQuality = quality;
	wcsncpy( lf.lfFaceName, description.c_str(), sizeof(lf.lfFaceName)/2-1 );
	lf.lfFaceName[ sizeof(lf.lfFaceName)/2-1 ] = 0;
	
	bool fontFound = false;
	EnumFontFamiliesExW( sic, &lf, (FONTENUMPROCW)ef_callback, (LPARAM)&fontFound, 0 );
	if (fontFound)
		font_handle = CreateFontIndirectW( &lf );

	if (font_handle)  
		SelectObject( sic, SelectObject( sic, font_handle ) ); //< Work around KB306198
	
	DeleteDC( sic );
}

bool font_renderer::ok() {
	return font_handle != 0;
}

font_renderer::~font_renderer() {
	if (font_handle)
		DeleteObject( font_handle );
}

void font_renderer::gl_render_to_texture( const view&, const wstring& text, layout_texture& tx ) {
	HDC dc = NULL;
	HBITMAP bmp = NULL;
	HFONT prevFont = NULL;

	try {
		dc = CreateCompatibleDC( NULL );

		prevFont = (HFONT)SelectObject( dc, font_handle );
		
		RECT rect;
		rect.left = 0;
		rect.top = 0;
		rect.right = 1024;
		rect.bottom = 1024;
		
		if (!DrawTextW( dc, text.c_str(), text.size(), &rect, DT_CALCRECT ))
			throw win32_exception("DrawText(DT_CALCRECT) failed.");
			
		if (!rect.right) rect.right = 1;
		if (!rect.bottom) rect.bottom = 1;

		BITMAPINFOHEADER hdr;
		memset(&hdr, 0, sizeof(hdr));
		hdr.biSize = sizeof(hdr);
		hdr.biWidth = rect.right;
		hdr.biHeight = rect.bottom;
		hdr.biPlanes = 1;
		hdr.biBitCount = 24;
		hdr.biCompression = BI_RGB;
		
		void* bits;
		bmp = CreateDIBSection( dc, (BITMAPINFO*)&hdr, DIB_RGB_COLORS, &bits, NULL, 0 );
		if (!bmp) throw win32_exception("CreateDIBSection failed.");
		
		int databytes = hdr.biWidth*3; // the actual BGR data
		int padbytes = (-hdr.biWidth*3)&3; // row is padded to have  multiple of 4 bytes
		int biPitch = databytes + padbytes;
		memset(bits, 0, biPitch * hdr.biHeight);
		
		SetTextColor(dc, 0xFFFFFF);
		SetBkColor(dc, 0);
		
		HBITMAP prevBmp = (HBITMAP)SelectObject( dc, bmp );
		DrawTextW( dc, text.c_str(), text.size(), &rect, 0 );
		SelectObject(dc, prevBmp);
		
		/*
		for (int row=0; row<hdr.biHeight; row++) {
			memset((char *)bits+(biPitch*row), 0xFF, 3);                // Clear left edge of texture
			memset((char *)bits+(biPitch*(row+1)-3-padbytes), 0, 3); // Clear right edge of texture
		}
		*/

		//memset((char *)bits, 0, biPitch);                          // Clear bottom edge of texture
		//memset((char *)bits+biPitch*(hdr.biHeight-1), 0, biPitch); // Clear top edge of texture

		SelectObject(dc, prevFont);
		DeleteDC( dc ); dc = NULL;
		
		tx.set_image( rect.right, -rect.bottom, isClearType ? GL_RGB : GL_LUMINANCE, GL_BGR_EXT, GL_UNSIGNED_BYTE, 4, bits );

		DeleteObject( bmp ); bmp = NULL;
	} catch( ... ) {
		if (bmp) DeleteObject( bmp );
		if (dc) { SelectObject(dc, prevFont); DeleteDC( dc ); }
		
		throw;
	}
}

} // namespace cvisual
