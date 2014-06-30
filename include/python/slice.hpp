#ifndef VPYTHON_PYTHON_SLICE_HPP
#define VPYTHON_PYTHON_SLICE_HPP

// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include <boost/python/object.hpp>
#include <boost/python/converter/pytype_object_mgr_traits.hpp>

#include <iterator>
#include <algorithm>

namespace cvisual { namespace python {

using boost::python::slice_nil;
namespace detail = boost::python::detail;
	
class slice : public boost::python::object
{
 public:
	// Equivalent to slice(::)
	slice();

	// Each argument must be int, slice_nil, or implicitly convertable to int
	template<typename Integer1, typename Integer2>
	slice( Integer1 start, Integer2 stop)
		: boost::python::object( boost::python::detail::new_reference( 
		 PySlice_New( object(start).ptr(), object(stop).ptr(), NULL)))
	{}
	
	template<typename Integer1, typename Integer2, typename Integer3>
	slice( Integer1 start, Integer2 stop, Integer3 stride)
		: boost::python::object( boost::python::detail::new_reference( 
			PySlice_New( object(start).ptr(), object(stop).ptr(), 
			object(stride).ptr())))
	{}
		
	// Get the Python objects associated with the slice.  In principle, these 
	// may be any arbitrary Python type, but in practice they are usually 
	// integers.  If one or more parameter is ommited in the Python expression 
	// that created this slice, than that parameter is None here, and compares 
	// equal to a default-constructed boost::python::object.
	// If a user-defined type wishes to support slicing, then support for the 
	// special meaning associated with negative indices is up to the user.
	boost::python::object start();
	boost::python::object stop();
	boost::python::object step();
		
 public:
	// This declaration, in conjunction with the specialization of 
	// object_manager_traits<> below, allows C++ functions accepting slice 
	// arguments to be called from Python.  These constructors should never
	// be used in client code.
	BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(slice, boost::python::object)
};
} } // !namespace cvisual::python

namespace boost { namespace python { namespace converter {

template<>
struct object_manager_traits<cvisual::python::slice>
	: pytype_object_manager_traits<&PySlice_Type, cvisual::python::slice>
{
};
	
} } }// !namespace boost::python::converter

#endif // !defined VVPYTHON_PYTHON_SLICE_HPP
