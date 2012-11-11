#ifndef VPYTHON_UTIL_TIMER_HPP
#define VPYTHON_UTIL_TIMER_HPP

// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

namespace cvisual {

class timer
{
 private:
	double start; ///< The system time at the last lap_start() call.
	double inv_tick_count;
 
 public:
	/** Construct a new timer. */
	timer();

	/** Time elapsed since timer was created. */
	double elapsed();
};

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_TIMER_HPP
