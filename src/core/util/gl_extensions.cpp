#include "util/gl_extensions.hpp"
#include "display_kernel.hpp"

namespace cvisual {

template <class PFN>
void getPFN( PFN& func, display_kernel& d, const char* name ) {
	func = reinterpret_cast<PFN>( d.getProcAddress( name ) );
	if (!func)
		throw std::runtime_error(
			("Unable to get extension function: " +
			(std::string)name + " even though the extension is advertised.").c_str() );
}

gl_extensions::gl_extensions() {
	memset( this, 0, sizeof(this) );
}

void gl_extensions::init( display_kernel& d ) {
	#define F( name ) getPFN( name, d, #name )

	if ( ARB_shader_objects = d.hasExtension( "GL_ARB_shader_objects" ) ) {
		F( glCreateProgramObjectARB );
		F( glLinkProgramARB );
		F( glUseProgramObjectARB );
		F( glCreateShaderObjectARB );
		F( glShaderSourceARB );
		F( glCompileShaderARB );
		F( glAttachObjectARB );
		F( glDeleteObjectARB );
		F( glGetHandleARB );
		F( glUniform1iARB );
		F( glUniformMatrix4fvARB );
		F( glUniform4fvARB );
		F( glGetUniformLocationARB );
		F( glGetObjectParameterivARB );
		F( glGetInfoLogARB );
	}

	if ( EXT_texture3D = d.hasExtension( "GL_EXT_texture3D" ) ) {
		F( glTexImage3D );
		F( glTexSubImage3D );
	}

	if ( ARB_multitexture = d.hasExtension( "GL_ARB_multitexture" ) ) {
		F( glActiveTexture );
	}

	if ( ARB_point_parameters = d.hasExtension( "GL_ARB_point_parameters" ) ) {
		F( glPointParameterfvARB );
	}
}

} // namespace cvisual
