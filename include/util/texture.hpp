#ifndef VPYTHON_UTIL_TEXTURE_HPP
#define VPYTHON_UTIL_TEXTURE_HPP

// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include <string>
#include <boost/shared_ptr.hpp>
#include "util/gl_free.hpp"
#include "wrap_gl.hpp"

namespace cvisual {

using boost::shared_ptr;

/** A class to assist in managing OpenGL texture resources.
*/
class texture
{
 private:
	bool damaged;
	unsigned int handle;

 public:
	/** Release the handle to OpenGL.  Subclasses must not call
 		glDeleteTextures() on this class's handle.
	*/
	virtual ~texture();

	/** True iff the texture object is managing something. */
	// operator bool() const;

	/** Make this texture active.  This function constitutes use under the
		"initialize on first use" rule, and will incur a one-time speed and
 		continuous graphics memory penalty.  Precondition: an OpenGL context
		must be active.
	*/
	void gl_activate(const struct view& scene);

	/** Determine whether or not this texture has an opacity channel.
		@returns True iff there is an opacity channel for this texture.
	*/
	bool has_opacity() const;

	/** Returns e.g. GL_TEXTURE_2D - the thing to be enabled to make this texture
	    work with the fixed function pipeline.
	*/
	virtual int enable_type() const;

 protected:
	// A unique identifier for the texture, to be obtained from glGenTextures().
	bool have_opacity;

	// Perform zero initialization of POD members.
	texture();

	// Make this class noncopyable.
	texture( const texture&);
	const texture& operator=( const texture&);

	// Sets handle and registers it to be freed at shutdown
	void set_handle( const view&, unsigned int handle );
	unsigned get_handle() { return handle; }

	// Called by gl_activate() on the first use and whenever damaged.
	// Postcondition: handle refers to an initialized OpenGL texture object.
	virtual void gl_init(const view&) = 0;

	// Perform any texture transformation matrix initialization that might be
	// required.  Default: do nothing.
	// This function must assume that the active matrix is GL_MODELVIEW and must
	// return in that state.
	virtual void gl_transform();

	// This function will be called by gl_activate() so that subclasses can
	// detect asynchronous changes to themselves
	virtual void damage_check();

	// Mutable subclasses must call this function whenever their texture data
	// needs to be reloaded into OpenGL.
	void damage();

 public:
 	// Should be protected; makeing this public works around a GCC 3.4.2 bug
	static void gl_free( GLuint handle );
};



#ifdef __GNUC__
#define PURE __attribute__((pure))
#else
#define PURE
#endif

size_t next_power_of_two(size_t arg) PURE ;

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_TEXTURE_HPP
