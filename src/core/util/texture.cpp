// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "wrap_gl.hpp"
#include "util/texture.hpp"
#include "util/errors.hpp"
#include <boost/lexical_cast.hpp>
#include <boost/bind.hpp>
using boost::lexical_cast;

namespace cvisual {

texture::texture()
	: damaged(false), handle(0), have_opacity(false)
{
}

texture::~texture()
{
	if (handle) on_gl_free.free( boost::bind( &gl_free, handle ) );
}

#if 0
texture::operator bool() const
{
	return handle != 0;
}
#endif

int
texture::enable_type() const {
	 return GL_TEXTURE_2D;
}

void
texture::gl_activate(const view& v)
{
	damage_check();
	if (damaged) {
		gl_init(v);
		damaged = false;
		check_gl_error();
	}
	if (!handle) return;
	
	glBindTexture( enable_type(), handle );
	this->gl_transform();
	check_gl_error();
}

bool
texture::has_opacity() const
{
	return have_opacity;
}

void
texture::set_handle( const view&, unsigned h ) {
	if (handle) on_gl_free.free( boost::bind( &gl_free, handle ) );

	handle = h;
	on_gl_free.connect( boost::bind(&texture::gl_free, handle) );
	VPYTHON_NOTE( "Allocated texture number " + lexical_cast<std::string>(handle));
}

void
texture::gl_free(GLuint handle)
{
	VPYTHON_NOTE( "Deleting texture number " + lexical_cast<std::string>(handle));
	glDeleteTextures(1, &handle);
}

void
texture::gl_transform()
{
}

void 
texture::damage_check()
{
}

void
texture::damage()
{
	damaged = true;
}

size_t
next_power_of_two(size_t arg) 
{
	size_t ret = 2;
	// upper bound of 28 chosen to limit memory growth to about 256MB, which is
	// _much_ larger than most supported textures
	while (ret < arg && ret < (1 << 28))
		ret <<= 1;
	return ret;
}

} // !namespace cvisual
