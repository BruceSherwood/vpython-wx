#ifndef VPYTHON_UTIL_RATE_HPP
#define VPYTHON_UTIL_RATE_HPP

#include <boost/python.hpp>
#include <boost/function.hpp>

namespace cvisual {

// This function is stateful and allows an application to control its loop
// execution rate.  When calling rate() once per iteration, rate inserts a small
// delay that is calibrated such that the loop will execute at about 'freq'
// iterations per second.
//void rate( const double& freq);

using namespace boost::python;

typedef void (wait_t)(void);

void set_wait(object obj);

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_RATE_HPP
