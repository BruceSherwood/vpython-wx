#ifndef VPYTHON_UTIL_SHADER_PROGRAM_HPP
#define VPYTHON_UTIL_SHADER_PROGRAM_HPP
#pragma once

#include "display_kernel.hpp"

#ifdef __APPLE__
// The following are needed in order to be able to query the rendering properties
#include <OpenGL/CGLCurrent.h>
#include <OpenGL/CGLTypes.h>
#include <OpenGL/OpenGL.h>
#endif

namespace cvisual {

class shader_program {
 public:
	shader_program( const std::string& source );
	~shader_program();
	
	const std::string& get_source() const { return source; }
	int get_uniform_location( const view& v, const char* name );
	void set_uniform_matrix( const view& v, int loc, const tmatrix& in );

 private:
	friend class use_shader_program;
	void realize( const view& );
	
	void compile( const view&, int type, const std::string& source );
	std::string getSection( const std::string& name );
	
	static void gl_free( PFNGLDELETEOBJECTARBPROC, int );
	
	std::string source;
	std::map<std::string, int> uniforms;
	GLhandleARB program;
	PFNGLDELETEOBJECTARBPROC glDeleteObjectARB;
};

class use_shader_program {
 public:
	// use_shader_program(NULL) does nothing, rather than enabling the fixed function
	//   pipeline explicitly.  This is convenient, but maybe we need a way to do the other thing?
	use_shader_program( const view& v, shader_program* program );
	use_shader_program( const view& v, shader_program& program );
	~use_shader_program();
	
	bool ok() { return m_ok; }  // true if the shader program was successfully invoked

 private:
	const view& v;
	GLhandleARB oldProgram;
	bool m_ok;
	void init( shader_program* program );
};

}

#endif
