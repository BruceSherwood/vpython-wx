#ifndef VPYTHON_UTIL_DISPLAYLIST_HPP
#define VPYTHON_UTIL_DISPLAYLIST_HPP

// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include <boost/shared_ptr.hpp>

namespace cvisual {

using boost::shared_ptr;

/** A manager for OpenGL displaylists */
class displaylist
{
 private:
	shared_ptr<class displaylist_impl> impl;
	bool built;
 
 public:
	displaylist();

	/** Begin compiling a new displaylist.  Nothing is drawn to the screen
 		when rendering commands into the displaylist.  Be sure to call 
 		gl_compile_end() when you are done.
 		*/
	void gl_compile_begin();
	
	/** Completes compiling the displaylist. */
	void gl_compile_end();
	
	/** Perform the OpenGL commands cached with gl_compile_begin() and 
		gl_compile_end(). */
	void gl_render() const;
	
	/** @return true iff this object contains a compiled OpenGL program. */
	//operator bool() const;
	bool compiled();
};

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_DISPLAYLIST_HPP
