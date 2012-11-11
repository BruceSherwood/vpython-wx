#include "util/gl_free.hpp"

namespace cvisual {

gl_free_manager on_gl_free;

boost::signal< void() >& gl_free_manager::on_shutdown() {
	static boost::signal< void() >* i = new boost::signal< void() >;
	return *i;
}

boost::signal< void() >& gl_free_manager::on_next_frame() {
	static boost::signal< void() >* i = new boost::signal< void() >;
	return *i;
}

void 
gl_free_manager::frame() {
	on_next_frame()();
	on_next_frame().disconnect_all_slots();
}

void
gl_free_manager::shutdown() {
	on_next_frame()();
	on_next_frame().disconnect_all_slots();
	on_shutdown()();
	on_shutdown().disconnect_all_slots();
}

}
