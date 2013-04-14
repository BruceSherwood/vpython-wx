// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/errors.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>

#include <boost/python/import.hpp>

namespace cvisual {

void write_stderr( const std::string& message ) {
	// TODO: Exception handling; in case of failure maybe print error to "real" stderr or a log file
	boost::python::import( "sys" ).attr( "stderr" ).attr( "write" )( message );
	boost::python::import( "sys" ).attr( "stderr" ).attr( "flush" )();
}

void
write_critical( 
	std::string file, 
	int line, 
	std::string function, 
	std::string message)
{
	std::ostringstream o;
	o << "VPython ***CRITICAL ERROR***: " << file << ":" << line << ": " 
	<< function << ": " << message << "\n";
	write_stderr( o.str() );
	return;
}

void
write_warning( 
	std::string file, 
	int line, 
	std::string function, 
	std::string message)
{
	std::ostringstream o;
	o << "VPython WARNING: " << file << ":" << line << ": " 
	<< function << ": " << message << "\n";
	write_stderr( o.str() );
	return;
}

void
write_note( std::string file, int line, std::string message)
{
	static bool enable = (bool)getenv( "VPYTHON_DEBUG");
	if (!enable) return;
	
	std::ostringstream o;
	o << "VPython: " << file << ":" << line << ": " << message 
		<< "\n";
	write_stderr( o.str() );
}

void
dump_glmatrix()
{
	// TODO: set this up to write out a matrix with the same format for all of
	// the members.
	float M[4][4];
	glGetFloatv( GL_MODELVIEW_MATRIX, M[0]);
	std::cout << "Modelview matrix status:\n";
	std::cout << "| " << M[0][0] << " " << M[1][0] << " " << M[2][0] << " " << M[3][0] << "|\n";
	std::cout << "| " << M[0][1] << " " << M[1][1] << " " << M[2][1] << " " << M[3][1] << "|\n";
	std::cout << "| " << M[0][2] << " " << M[1][2] << " " << M[2][2] << " " << M[3][2] << "|\n";
	std::cout << "| " << M[0][3] << " " << M[1][3] << " " << M[2][3] << " " << M[3][3] << "|\n";
	
	glGetFloatv( GL_PROJECTION_MATRIX, M[0]);
	std::cout << "Projection matrix status:\n";
	std::cout << "| " << M[0][0] << " " << M[1][0] << " " << M[2][0] << " " << M[3][0] << "|\n";
	std::cout << "| " << M[0][1] << " " << M[1][1] << " " << M[2][1] << " " << M[3][1] << "|\n";
	std::cout << "| " << M[0][2] << " " << M[1][2] << " " << M[2][2] << " " << M[3][2] << "|\n";
	std::cout << "| " << M[0][3] << " " << M[1][3] << " " << M[2][3] << " " << M[3][3] << "|\n";	
}

void
clear_gl_error_real()
{
	#ifndef NDEBUG
	for(GLenum err_code = glGetError(); err_code != GL_NO_ERROR; err_code = glGetError());
	#endif
}

void
check_gl_error_real( const char* file, int line)
{
	#ifndef NDEBUG
	int errcount = 0;
	GLenum err_code, firsterr;
	for(err_code = glGetError(); err_code != GL_NO_ERROR; err_code = glGetError())
	{
		if (!errcount) firsterr = err_code;
		++errcount;
	}
	if (errcount) {
		std::ostringstream err;
		// Insert the manual cast from the unsigned char pointer to signed char pointer type.
		err << file << ":" << line << " " << (const char*)gluErrorString(firsterr);
		throw gl_error( err.str().c_str(), err_code);
	}
	#endif
}

gl_error::gl_error( const char* msg, const GLenum err)
	: std::runtime_error(msg), error( err)
{
}

gl_error::gl_error( const char* msg)
	: std::runtime_error(msg), error( GL_NO_ERROR)
{
}

} // !namespace cvisual
