// This file takes roughly 115 MB RAM to compile.

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include <stdexcept>
#include <exception>
#include <iostream>

#include <boost/python/exception_translator.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/def.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL visual_PyArrayHandle
//#include <numpy/arrayobject.h>

#include "util/rate.hpp"
#include "util/errors.hpp"
#include "python/num_util.hpp"

// Python 2/3 compatibility
#ifndef PyString_Check
#define PyString_Check   PyUnicode_Check
#endif

namespace cvisual {
void wrap_display_kernel();
void wrap_primitive();
void wrap_rgba();
void wrap_vector();
void wrap_arrayobjects();

void
translate_std_out_of_range( std::out_of_range e)
{
	PyErr_SetString( PyExc_IndexError, e.what());
}

void
translate_std_invalid_argument( std::invalid_argument e)
{
	PyErr_SetString( PyExc_ValueError, e.what());
}

void
translate_std_runtime_error( std::runtime_error e)
{
	PyErr_SetString( PyExc_RuntimeError, e.what());
}

namespace py = boost::python;

struct double_from_int
{
	double_from_int()
	{
		py::converter::registry::push_back(
			&convertible,
			&construct,
			py::type_id<double>());
	}

	static void* convertible( PyObject* obj)
	{
		PyObject* newobj = PyNumber_Float(obj);
		if (!PyString_Check(obj) && newobj) {
			Py_DECREF(newobj);
			return obj;
		} else {
			if (newobj) {
				Py_DECREF(newobj);
			}
			PyErr_Clear();
			return 0;
		}
	}

	static void construct(
		PyObject* _obj,
		py::converter::rvalue_from_python_stage1_data* data)
	{
		PyObject* newobj = PyNumber_Float(_obj);
        double* storage = (double*)(
            (py::converter::rvalue_from_python_storage<double>*)
            data)->storage.bytes;
        *storage = py::extract<double>(newobj);
		Py_DECREF(newobj);
		data->convertible = storage;
	}
};

struct float_from_int
{
	float_from_int()
	{
		py::converter::registry::push_back(
			&convertible,
			&construct,
			py::type_id<float>());
	}

	static void* convertible( PyObject* obj)
	{
		PyObject* newobj = PyNumber_Float(obj);
		if (!PyString_Check(obj) && newobj) {
			Py_DECREF(newobj);
			return obj;
		} else {
			if (newobj) {
				Py_DECREF(newobj);
			}
			PyErr_Clear();
			return 0;
		}
	}

	static void construct(
		PyObject* _obj,
		py::converter::rvalue_from_python_stage1_data* data)
	{
		PyObject* newobj = PyNumber_Float(_obj);
        float* storage = (float*)(
            (py::converter::rvalue_from_python_storage<float>*)
            data)->storage.bytes;
        *storage = py::extract<float>(newobj);
		Py_DECREF(newobj);
		data->convertible = storage;
	}
};

BOOST_PYTHON_MODULE( cvisual)
{
	VPYTHON_NOTE( "Importing cvisual from vpython-core2.");

	using namespace boost::python;
	numeric::array::set_module_and_type( "numpy", "ndarray");

#if __GNUG__
#if __GNUC__ == 3
#if __GNUCMINOR__ >= 1 && __GNUCMINOR__ < 4
	std::set_terminate( __gnu_cxx::__verbose_terminate_handler);
#endif
#endif
#endif

	// A subset of the python standard exceptions may be thrown from visual
	register_exception_translator<std::out_of_range>(
		&translate_std_out_of_range);
	register_exception_translator<std::invalid_argument>(
		&translate_std_invalid_argument);
	register_exception_translator<std::runtime_error>(
		&translate_std_runtime_error);

	def("set_wait", set_wait);

	double_from_int();
	float_from_int();
	wrap_vector();
	wrap_rgba();
	wrap_display_kernel();
	wrap_primitive();
	wrap_arrayobjects();
	python::init_numpy(); // initialize numpy
}

} // !namespace cvisual
