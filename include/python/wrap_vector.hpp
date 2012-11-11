#ifndef CVISUAL_PYTHON_WRAP_VECTOR_HPP
#define CVISUAL_PYTHON_WRAP_VECTOR_HPP

#include "util/vector.hpp"
#include <boost/python/object.hpp>

namespace cvisual {
 
// Convert a general Python object to a Visual vector.  This function may
// throw.
cvisual::vector tovector( boost::python::object);

} // !namespace cvisual::python

#endif
