#ifndef VPYTHON_UTIL_RATE_HPP
#define VPYTHON_UTIL_RATE_HPP

#include <boost/python.hpp>
#include <boost/function.hpp>
#include "util/rgba.hpp"

namespace cvisual {

using namespace boost::python;

typedef void (wait_t)(double);

void set_wait(object obj);

/*
typedef void (text_to_bitmap_t)(std::wstring& text,
		rgb& foreground, rgb& background,
		double height, std::wstring& font,
		std::wstring& style, std::wstring& weight);

void set_text_to_bitmap(object obj);
*/

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_RATE_HPP
