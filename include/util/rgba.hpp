#ifndef VPYTHON_UTIL_RGBA_HPP
#define VPYTHON_UTIL_RGBA_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "wrap_gl.hpp"

namespace cvisual {

/** A helper class to manage OpenGL color attributes.  The data is layout
	compatable with OpenGL's needs for the various vector forms of commands,
	like glColor4fv(), and glColorPointer().
*/
class rgba
{
 public:
	/** Red channel intensity, clamped to [0,1] */
	float red;
	/** Green channel intensity, clamped to [0,1] */
	float green;
	/** Blue channel intensity, clamped to [0,1] */
	float blue;
	/** Opacity, clamped to [0,1] */
	float opacity;

	/** Defaults to opaque white. */
	inline rgba() : red(1.0), green(1.0), blue(1.0), opacity(1.0) {}
	/** Allocate a new color. */
	inline rgba( float r, float g, float b, float a = 1.0)
		: red(r), green(g), blue(b), opacity(a) {}
	inline explicit rgba( const float* c)
		: red(c[0]), green(c[1]), blue(c[2]), opacity( c[3]) {}

	/** Convert to HSVA, lower saturation by 50%, convert back to RGBA.
		@return The desaturated color.
	*/
	rgba desaturate() const;
	/** Convert to greyscale, accounting for differences in perception.  This
		function makes 4 calls to std::pow(), and is very slow.
		@return The scaled color.
	*/
	rgba grayscale() const;

	/** Make this the active OpenGL color using glColor(). */
	inline void gl_set() const
	{ glColor4fv( &red); }
};


class rgb
{
 public:
	float red;
	float green;
	float blue;

	inline rgb() : red(1.0f), green(1.0f), blue(1.0f) {}
	inline rgb( float r, float g, float b)
		: red(r), green(g), blue(b)
	{}
	inline explicit rgb( float bw)
		: red(bw), green(bw), blue(bw)
	{}
	inline explicit rgb( const float* c)
		: red(c[0]), green(c[1]), blue(c[2]) {}
	inline explicit rgb( const double* c)
		: red(c[0]), green(c[1]), blue(c[2]) {}
	inline rgb( const rgb& other)
		: red( other.red), green( other.green), blue(other.blue)
	{}
	inline operator rgb() const { return rgb( red, green, blue); }

	rgb desaturate() const;
	rgb grayscale() const;

	float operator[](int i) const { return (&red)[i]; }

	inline void gl_set(float opacity) const
	{ glColor4f( red, green, blue, opacity); }
};

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_RGBA_HPP
