
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/rgba.hpp"

#include <boost/python/to_python_converter.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/proxy.hpp>
#include <boost/python/class.hpp>

// Python 2/3 compatibility
#ifndef PyInt_Check
#define PyInt_Check   PyLong_Check
#endif

namespace cvisual {
namespace py = boost::python;

struct rgb_from_seq
{
	rgb_from_seq()
	{
		py::converter::registry::push_back(
			&convertible,
			&construct,
			py::type_id<rgb>());
	}

	static void* convertible( PyObject* obj)
	{
		using py::handle;
		using py::allow_null;

		handle<> obj_iter( allow_null( PyObject_GetIter(obj)));
		if (!obj_iter.get()) {
			PyErr_Clear();
			return 0;
		}
		int obj_size = PyObject_Length(obj);
		if (obj_size < 0) {
			PyErr_Clear();
			return 0;
		}
		if (obj_size != 3)
			return 0;
		return obj;
	}

	static void construct(
		PyObject* _obj,
		py::converter::rvalue_from_python_stage1_data* data)
	{
		py::object obj = py::object(py::handle<>(py::borrowed(_obj)));
		void* storage = (
			(py::converter::rvalue_from_python_storage<rgb>*)
			data)->storage.bytes;
		new (storage) rgb(
			py::extract<float>(obj[0]),
			py::extract<float>(obj[1]),
			py::extract<float>(obj[2]));
		data->convertible = storage;
	}
};

struct rgba_from_seq
{
	rgba_from_seq()
	{
		py::converter::registry::push_back(
			&convertible,
			&construct,
			py::type_id<rgba>());
	}

	static void* convertible( PyObject* obj)
	{
		using py::handle;
		using py::allow_null;
		if (PyInt_Check(obj) || PyFloat_Check(obj))
			return obj;

		handle<> obj_iter( allow_null( PyObject_GetIter(obj)));
		if (!obj_iter.get()) {
			PyErr_Clear();
			return 0;
		}
		int obj_size = PyObject_Length(obj);
		if (obj_size < 0) {
			PyErr_Clear();
			return 0;
		}
		if (obj_size != 3 && obj_size != 4)
			return 0;
		return obj;
	}

	static void construct(
		PyObject* _obj,
		py::converter::rvalue_from_python_stage1_data* data)
	{
		py::object obj = py::object(py::handle<>(py::borrowed(_obj)));
		void* storage = (
			(py::converter::rvalue_from_python_storage<rgba>*)
			data)->storage.bytes;
		int obj_size = PyObject_Length(_obj);
		if (obj_size == 3)
			new (storage) rgba(
				py::extract<float>(obj[0]),
				py::extract<float>(obj[1]),
				py::extract<float>(obj[2]));
		else
			new (storage) rgba(
				py::extract<float>(obj[0]),
				py::extract<float>(obj[1]),
				py::extract<float>(obj[2]),
				py::extract<float>(obj[3]));

		data->convertible = storage;
	}
};

struct rgb_to_tuple
{
	static PyObject* convert( const rgb& color)
	{
		py::tuple ret = py::make_tuple( color.red, color.green, color.blue);
		Py_INCREF(ret.ptr());
		return ret.ptr();
	}
};

struct rgba_to_tuple
{
	static PyObject* convert( const rgba& color)
	{
		py::tuple ret = py::make_tuple( color.red, color.green, color.blue, color.opacity);
		Py_INCREF(ret.ptr());
		return ret.ptr();
	}
};

void
wrap_rgba()
{
	rgb_from_seq();
	rgba_from_seq();
	py::to_python_converter< rgb, rgb_to_tuple>();
	py::to_python_converter< rgba, rgba_to_tuple>();
}

} // !namespace cvisual
