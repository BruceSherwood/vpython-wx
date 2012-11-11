// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/displaylist.hpp"
#include "util/gl_free.hpp"
#include "util/errors.hpp"
#include "wrap_gl.hpp"
#include <cassert>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
using boost::lexical_cast;

namespace cvisual {

class displaylist_impl : boost::noncopyable {
 private:
	unsigned int handle;

	static void gl_free(unsigned int handle) {
		glDeleteLists( handle, 1 );
	}
	
 public:
	displaylist_impl() {
		handle = glGenLists(1);
		on_gl_free.connect( boost::bind(&displaylist_impl::gl_free, handle) );
		glNewList( handle, GL_COMPILE );
	}
	~displaylist_impl() {
		compile_end();
		on_gl_free.free( boost::bind(&displaylist_impl::gl_free, handle) );
	}
	
	void compile_end() {
		glEndList();
	}
	
	void call() {
		glCallList( handle );
	}
	
	//operator bool() { return handle && built; }
};

// sphere.cpp, pyramid.cpp

displaylist::displaylist() : built(false) {}

void 
displaylist::gl_compile_begin()
{
	impl.reset( new displaylist_impl );
}
	
void 
displaylist::gl_compile_end()
{
	impl->compile_end();
	built = true;
}

void
displaylist::gl_render() const
{
	impl->call();
}

//displaylist::operator bool() const {
bool
displaylist::compiled() {
	return built;
}

} // !namespace cvisual
