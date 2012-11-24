#ifndef VPYTHON_UTIL_RATE_HPP
#define VPYTHON_UTIL_RATE_HPP

#include <boost/python.hpp>
#include <boost/function.hpp>

namespace cvisual {

using namespace boost::python;

typedef void (wait_t)(double);

void set_wait(object obj);

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_RATE_HPP
