#ifndef VPYTHON_UTIL_GL_ENABLE_HPP
#define VPYTHON_UTIL_GL_ENABLE_HPP

#include "wrap_gl.hpp"

namespace cvisual {

// Stack-unwind safe versions of gl{Enable,Disable}{ClientState,}()
class gl_enable
{
 private:
	GLenum value;
 public:
	inline gl_enable( GLenum v) : value(v)
	{ glEnable( value); }

	inline ~gl_enable()
	{ glDisable( value); }
};

class gl_enable_client
{
 private:
	GLenum value;
 public:
	inline gl_enable_client( GLenum v)	: value(v)
	{ glEnableClientState( value); }

	inline ~gl_enable_client()
	{ glDisableClientState( value); }
};

class gl_disable
{
 private:
	GLenum value;
 public:
	inline gl_disable( GLenum v) : value(v)
	{ glDisable( value); }

	inline ~gl_disable()
	{ glEnable( value); }
};
	
} // !namespace cvisual

#endif /*VPYTHON_UTIL_GL_ENABLE_HPP*/
