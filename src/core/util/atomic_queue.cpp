#include "util/atomic_queue.hpp"

#include <boost/python/errors.hpp>
#include <iostream>
#include "util/errors.hpp"

namespace cvisual {

void
atomic_queue_impl::push_notify()
{
	empty = false;
}

} // !namespace cvisual
