#ifndef VPYTHON_UTIL_GL_FREE_HPP
#define VPYTHON_UTIL_GL_FREE_HPP

#include <boost/signals.hpp>

namespace cvisual {

class gl_free_manager {
 public:
	// The callback will be called if the OpenGL context(s) are destroyed
	template <class T>
	void connect( T callback ) { on_shutdown().connect( callback ); }
	
	// The callback will be called the next time OpenGL objects may be freed, and
	//   will no longer be called on shutdown().
	template <class T>
	void free( T callback ) { on_next_frame().connect( callback ); on_shutdown().disconnect( callback ); }
	
	// Call with OpenGL context active
	void frame();
	
	// Call when the context(s) are destroyed.  The gl_free_manager can be reused.
	void shutdown();

 private:
	boost::signal< void() > &on_shutdown();
	boost::signal< void() > &on_next_frame();
};

// At present, there is just one of these, because all OpenGL contexts share server
// objects through wglShareLists or equivalent.  It is shutdown when all contexts are
// shut down.  If this design is changed in the
// future, there will need to be an instance for each context.
extern gl_free_manager on_gl_free;

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_GL_FREE_HPP
