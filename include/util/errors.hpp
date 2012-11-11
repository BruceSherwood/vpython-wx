#ifndef VPYTHON_UTIL_ERRORS_HPP
#define VPYTHON_UTIL_ERRORS_HPP

// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.


//AS changed __PRETTY_FUNCTION to __FUNCTION__ for VC++ compatibility

#include "wrap_gl.hpp"

#include <string>
#include <stdexcept>

namespace cvisual {

/** Report the existance of a critical error to cerr. */
#define VPYTHON_CRITICAL_ERROR(msg) write_critical( __FILE__, __LINE__, \
	__FUNCTION__, msg)
/** Report a warning to the user that can probably be corrected, through cerr. */
#define VPYTHON_WARNING(msg) write_warning( __FILE__, __LINE__, \
	__FUNCTION__, msg)

#define VPYTHON_NOTE(msg) write_note( __FILE__, __LINE__, msg)

// This should only be used within Win32-specific code.
#define WIN32_CRITICAL_ERROR(msg) win32_write_critical( __FILE__, __LINE__, \
	__FUNCTION__, msg)

void
win32_write_critical(
	std::string file, int line, std::string func, std::string msg);

void
write_critical(
	std::string file, int line, std::string function, std::string message);

void
write_warning(
	std::string file, int line, std::string function, std::string message);

void
write_note( std::string file, int line, std::string message);

void
write_stderr( const std::string& message );

/** Obtains the active OpenGL transformation matrix and dumps it to stderr. */
void
dump_glmatrix();

/** Clears the OpenGL error state.  If NDEBUG is set, this function is a no-op. */
void
clear_gl_error_real( void);

/** Checks the OpenGL error state and throws gl_error if it is anything other
than GL_NO_ERROR.  If NDEBUG is set, this function is a no-op.
 */
void
check_gl_error_real(const char* file, int line);

// Forward the call to the real function.
#ifdef NDEBUG
# define check_gl_error() do {} while (false)
# define clear_gl_error() do {} while (false)
#else
# define check_gl_error() check_gl_error_real(__FILE__, __LINE__)
# define clear_gl_error() clear_gl_error_real()
#endif

/** Exception class thrown by check_gl_error() */
class gl_error : public std::runtime_error
{
 private:
	GLenum error;
 public:
	/** Returns the OpenGL error code that triggered this exception. */
	inline GLenum
	get_error_code() const
	{ return error; }

	/** Construct an error in preparation to throw it.
		@param msg A human-readable error message that will be used for what().
		@param code The triggering OpenGL error code.
		*/
	gl_error( const char* msg, const GLenum code);
	/** Construct an error in preparation to throw it.  This should not be used.
		@param msg A human-readable error message that will be used for what().
		*/
	gl_error( const char* msg);
};

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_ERRORS_HPP
