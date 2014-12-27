// Copyright (c) 2006 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "python/numeric_texture.hpp"
#include "util/gl_enable.hpp"
#include "util/errors.hpp"
#include "renderable.hpp"

#include <boost/bind.hpp>
#include <boost/crc.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_array.hpp>
#include <iostream>

namespace cvisual { namespace python {


namespace {

GLenum
gl_type_name( NPY_TYPES t)
{
	switch (t) {
	    case NPY_BYTE:
	    	return GL_BYTE;
	    case NPY_UBYTE:
	    	return GL_UNSIGNED_BYTE;
	    case NPY_SHORT:
	    	return GL_SHORT;
	    case NPY_INT:
	    	return GL_INT;
	    case NPY_FLOAT:
	    	return GL_FLOAT;
	    default:
	    	return 1;
	}
}

boost::crc_32_type engine;
} // !namespace (anonymous)

numeric_texture::numeric_texture()
	: texdata(0),
	data_width(0), data_height(0), data_depth(0), data_channels(0), data_type(NPY_NOTYPE),
		data_textype( 0), data_mipmapped(true), data_antialias(false), data_clamp(false),
	tex_width(0), tex_height(0), tex_depth(0), tex_channels(0), tex_type(NPY_NOTYPE),
		tex_textype( 0), tex_mipmapped(true), tex_antialias(false), tex_clamp(false)
{
}

numeric_texture::~numeric_texture()
{
}

int
numeric_texture::enable_type() const
{
	return data_depth ? GL_TEXTURE_3D_EXT : GL_TEXTURE_2D;
}

bool
numeric_texture::degenerate() const
{
	return data_width == 0 || data_height == 0 || data_channels == 0 || data_type == NPY_NOTYPE;
}

bool
numeric_texture::should_reinitialize(void) const
{
	return (
		data_channels != tex_channels ||
		data_mipmapped != tex_mipmapped ||
		data_clamp != tex_clamp ||
		data_type != tex_type ||
		(
			!tex_mipmapped &&
			(next_power_of_two(data_width) != tex_width ||
			next_power_of_two(data_height) != tex_height ||
			next_power_of_two(data_depth) != tex_depth)
		) ||
		(
			tex_mipmapped &&
			(data_width != tex_width ||
			data_height != tex_height ||
			data_depth != tex_depth)
		)
	);
}

void
numeric_texture::gl_init( const view& v )
{
	if (degenerate())
		return;

	int type = data_depth ? GL_TEXTURE_3D_EXT : GL_TEXTURE_2D;

	if (type == GL_TEXTURE_3D_EXT && !v.glext.EXT_texture3D)
		return;

	GLuint handle = get_handle();
	if (!handle) {
		glGenTextures(1, &handle);
		set_handle( v, handle );
	}
	glBindTexture(type, handle);

	if (data_mipmapped) {
		glTexParameteri( type, GL_TEXTURE_MIN_FILTER,
			data_antialias ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST);
		glTexParameteri( type, GL_TEXTURE_MAG_FILTER,
			data_antialias ? GL_LINEAR : GL_NEAREST);
	}
	else {
		glTexParameteri( type, GL_TEXTURE_MIN_FILTER,
			data_antialias ? GL_LINEAR : GL_NEAREST);
		glTexParameteri( type, GL_TEXTURE_MAG_FILTER,
			data_antialias ? GL_LINEAR : GL_NEAREST);
	}
	tex_antialias = data_antialias;
	glTexParameteri( type, GL_TEXTURE_WRAP_S, data_clamp ? GL_CLAMP : GL_REPEAT );
	glTexParameteri( type, GL_TEXTURE_WRAP_T, data_clamp ? GL_CLAMP : GL_REPEAT );
	glTexParameteri( type, GL_TEXTURE_WRAP_R, data_clamp ? GL_CLAMP : GL_REPEAT );
	tex_clamp = data_clamp;
	check_gl_error();

	// Something is damaged.  Either the texture must be reinitialized
	// or just its data has changed.
	bool reinitialize = should_reinitialize();

	GLenum internal_format;
	if (!data_textype) {
		switch (data_channels) {
			case 1:
				internal_format = GL_LUMINANCE;
				break;
			case 2:
				internal_format = GL_LUMINANCE_ALPHA;
				break;
			case 3:
				internal_format = GL_RGB;
				break;
			case 4:
				internal_format = GL_RGBA;
				break;
			default: // Won't ever happen
				internal_format = GL_RGB;
		}
	}
	else {
		internal_format = data_textype;

		switch (data_textype) {
			case GL_LUMINANCE:
				if (data_channels != 1)
					throw std::invalid_argument(
						"Specify luminance data with single values.");
				break;
			case GL_ALPHA:
				if (data_channels != 1)
					throw std::invalid_argument(
						"Specify opacity data with single values.");
				break;
			case GL_LUMINANCE_ALPHA:
				if (data_channels != 2)
					throw std::invalid_argument(
						"Specify luminance and opacity data with double values, [luminance,opacity].");
				break;
			case GL_RGB:
				if (data_channels != 3)
					throw std::invalid_argument(
						"Specify RGB data with triple values, [r,g,b].");
				break;
			case GL_RGBA:
				if (data_channels != 4)
					throw std::invalid_argument(
						"Specify RGB_opacity data with quadruple values, [r,g,b,opacity].");
				break;
			case 0: default: // Won't ever happen
				break;
		}
	}
	tex_textype = internal_format;

	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );

	if (data_mipmapped && !data_depth) {
		tex_width = data_width;
		tex_height = data_height;
		tex_depth = data_depth;
		tex_channels = data_channels;
		tex_type = data_type;
		tex_textype = data_textype;
		tex_mipmapped = true;

		gluBuild2DMipmaps( type, internal_format, tex_width, tex_height,
			internal_format, gl_type_name(tex_type), data(texdata));
	} else {
		if (reinitialize) {
			tex_width = next_power_of_two(data_width);
			tex_height = next_power_of_two(data_height);
			tex_depth = data_depth ? next_power_of_two(data_depth) : 1;
			tex_channels = data_channels;
			tex_textype = data_textype;
			tex_type = data_type;
			tex_mipmapped = false;

			#ifdef __APPLE__
				// Work around a bug in macbook pro nvidia drivers' glTexSubImage3D
				//  GL_TEXTURE_STORAGE_HINT_APPLE = GL_STORAGE_CACHED_APPLE
				//  See http://www.mailinglistarchive.com/mac-opengl@lists.apple.com/msg03035.html
				glTexParameteri(GL_TEXTURE_3D, 0x85BC, 0x85BE);
			#endif

			if (type == GL_TEXTURE_3D_EXT) {
				v.glext.glTexImage3D( type, 0, internal_format, tex_width, tex_height, tex_depth,
					0, internal_format, gl_type_name(tex_type), NULL );
			} else {
				glTexImage2D( type, 0, internal_format, tex_width, tex_height,
					0, internal_format, gl_type_name( tex_type), NULL );
			}
		}

		if (type == GL_TEXTURE_3D_EXT) {
			v.glext.glTexSubImage3D( type, 0,
				0, 0, 0, data_width, data_height, data_depth,
				internal_format, gl_type_name(tex_type), data(texdata));
		} else {
			glTexSubImage2D( type, 0,
				0, 0, data_width, data_height,
				internal_format, gl_type_name(tex_type), data(texdata));
		}
	}

	check_gl_error();
}

void
numeric_texture::gl_transform(void)
{
	if (degenerate())
		return;
	glMatrixMode( GL_TEXTURE);
	glLoadIdentity();
	if (data_width != tex_width || data_height != tex_height) {
		float x_scale = float(data_width) / tex_width;
		float y_scale = float(data_height) / tex_height;
		glScalef( x_scale, y_scale, 1);
	}
	glMatrixMode( GL_MODELVIEW);
}

void
numeric_texture::set_data( boost::python::numeric::array data)
{
	namespace py = boost::python;

	NPY_TYPES t = type(data);
	if (t == NPY_CFLOAT || t == NPY_CDOUBLE || t == NPY_OBJECT || t == NPY_NOTYPE)
		throw std::invalid_argument( "Invalid texture data type");

	std::vector<npy_intp> dims = shape( data);
	if (dims.size() < 2 || dims.size() > 4) {
		throw std::invalid_argument( "Texture data must be NxMxC or NxM (or NxMxZxC for volume texture)");
	}

	if (t == NPY_DOUBLE) {
		data = astype( data, NPY_FLOAT);
		t = NPY_FLOAT;
	} else if (t == NPY_LONG) {
		data = astype( data, NPY_INT);
		t = NPY_INT;
	} else {
		// Make a copy, so the user can't mutate the texture in place (it's just too expensive
		// to check for changes; we make the user assign to texture.data again)
		data = py::extract<py::numeric::array>( data.copy() );
	}

	int channels = dims.size() >= 3 ? dims.back() : 1;
	if (channels < 1 || channels > 4) {
		throw std::invalid_argument(
			"Texture data must be NxMxC, where C is between 1 and 4 (inclusive)");
	}

	damage();
	texdata = data;
	data_width = dims[1];
	data_height = dims[0];
	if (dims.size() == 4) data_depth = dims[2]; else data_depth = 0;
	data_channels = channels;
	have_opacity = (
		data_channels == 2 ||
		data_channels == 4 ||
		(data_channels == 1 && data_textype == GL_ALPHA)
	);

	data_type = t;
}

boost::python::numeric::array
numeric_texture::get_data()
{
	// Return a copy, so the user can't mutate the texture in place (it's just too expensive
	// to check for changes; we make the user assign to texture.data again)
	return boost::python::extract< boost::python::numeric::array >( texdata.copy() );
}

void
numeric_texture::set_type( std::string requested_type)
{
	GLenum req_type = 0;
	if (requested_type == "luminance")
		req_type = GL_LUMINANCE;
	else if (requested_type == "opacity")
		req_type = GL_ALPHA;
	else if (requested_type == "luminance_opacity")
		req_type = GL_LUMINANCE_ALPHA;
	else if (requested_type == "rgb")
		req_type = GL_RGB;
	else if (requested_type == "rgbo")
		req_type = GL_RGBA;
	else if (requested_type == "auto")
		req_type = 0;
	else
		throw std::invalid_argument( "Unknown texture type");

	data_textype = req_type;
	if (req_type == GL_RGBA || req_type == GL_ALPHA || req_type == GL_LUMINANCE_ALPHA)
		have_opacity = true;
	damage();
}

std::string
numeric_texture::get_type() const
{
	switch (data_textype) {
		case GL_LUMINANCE:
			return std::string( "luminance");
		case GL_ALPHA:
			return std::string( "opacity");
		case GL_LUMINANCE_ALPHA:
			return std::string( "luminance_opacity");
		case GL_RGB:
			return std::string( "rgb");
		case GL_RGBA:
			return std::string( "rgbo");
		case 0: default:
			return std::string( "auto");
	}
}

void
numeric_texture::set_mipmapped( bool m)
{
	damage();
	data_mipmapped = m;
}

bool
numeric_texture::is_mipmapped(void)
{
	return data_mipmapped;
}

void
numeric_texture::set_antialias( bool aa)
{
	damage();
	data_antialias = aa;
}

bool
numeric_texture::is_antialiased( void)
{
	return data_antialias;
}

} } // !namespace cvisual::python
