// This file currently requires 137 MB to compile (optimizing).

// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/vector.hpp"

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/init.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/to_python_converter.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/extract.hpp>
#include "python/num_util.hpp"



namespace cvisual {
namespace py = boost::python;
using namespace cvisual::python;

using py::numeric::array;
using py::object;
using py::extract;

//AS add
using py::allow_null;

// Operations on Numeric arrays
namespace {

void
validate_array( const array& arr)
{
	std::vector<npy_intp> dims = shape(arr);
	if (type(arr) != NPY_DOUBLE) {
		throw std::invalid_argument( "Array must be of type Float64.");
	}
	if (!iscontiguous(arr)) {
		throw std::invalid_argument( "Array must be contiguous."
			"(Did you pass a slice?)");
	}
	if (dims.size() != 2) {
		if (dims.size() == 1 && dims[0] == 3)
			return;
		else
			throw std::invalid_argument( "Array must be Nx3 in shape.");
	}
	if (dims[1] != 3) {
		throw std::invalid_argument( "Array must be Nx3 in shape.");
	}
}

// Numeric doens't support the Sequence protocol, so I have to use this hack
// instead.
// 2008/2/16 BAS asks, "What is the situation with numpy?" Should look into this.
inline int
length(boost::python::object seq)
{
	int ret = PySequence_Size( seq.ptr());
	if (ret == -1) {
		boost::python::throw_error_already_set();
	}
	return ret;
}

} // !namespace anonymous

vector
tovector( py::object arr)
{
	switch (length(arr)) {
		case 2:
			return vector(
				extract<double>(arr[0]),
				extract<double>(arr[1]));
		case 3:
			return vector(
				extract<double>(arr[0]),
				extract<double>(arr[1]),
				extract<double>(arr[2]));
		default:
			throw std::invalid_argument("Vectors must have length 2 or 3");
	}
}

object
mag_a( const array& arr)
{
	validate_array( arr);
	std::vector<npy_intp> dims = shape(arr);
	// Magnitude of a flat 3-length array
	if (dims.size() == 1 && dims[0] == 3) {
		return object( vector(
			extract<double>(arr[0]),
			extract<double>(arr[1]),
			extract<double>(arr[2])).mag());
	}
	std::vector<npy_intp> rdims(1);
	rdims[0] = dims[0];
	array ret = makeNum( rdims);
	for (int i = 0; i< rdims[0]; ++i) {
		ret[i] = tovector(arr[i]).mag();
	}
	return ret;
}

object
mag2_a( const array& arr)
{
	validate_array( arr);
	std::vector<npy_intp> dims = shape(arr);
	if (dims.size() == 1 && dims[0] == 3) {
		// Returns an object of type float.
		return object( vector(
			extract<double>(arr[0]),
			extract<double>(arr[1]),
			extract<double>(arr[2])).mag2());
	}
	std::vector<npy_intp> rdims(1);
	rdims[0] = dims[0];
	array ret = makeNum( rdims);
	for (int i = 0; i < rdims[0]; ++i) {
		ret[i] = tovector(arr[i]).mag2();
	}
	// Returns an object of type Numeric.array.
	return ret;
}

object
norm_a( const array& arr)
{
	validate_array( arr);
	std::vector<npy_intp> dims = shape(arr);
	if (dims.size() == 1 && dims[0] == 3) {
		// Returns a float
		return object( vector(
			extract<double>(arr[0]),
			extract<double>(arr[1]),
			extract<double>(arr[2])).norm());
	}
	array ret = makeNum(dims);
	for (int i = 0; i < dims[0]; ++i) {
		ret[i] = tovector(arr[i]).norm();
	}
	// Returns a Numeric.array
	return ret;
}

array
dot_a( const array& arg1, const array& arg2)
{
	validate_array( arg1);
	validate_array( arg2);
	std::vector<npy_intp> dims1 = shape(arg1);
	std::vector<npy_intp> dims2 = shape(arg2);
	if (dims1 != dims2) {
		throw std::invalid_argument( "Array shape mismatch.");
	}

	std::vector<npy_intp> dims_ret(1);
	dims_ret[0] = dims1[0];
	array ret = makeNum( dims_ret);
	const double* arg1_i = (double*)data(arg1);
	const double* arg2_i = (double*)data(arg2);
	for ( int i = 0; i < dims1[0]; ++i, arg1_i +=3, arg2_i += 3) {
		ret[i] = vector(arg1_i).dot( vector(arg2_i));
	}
	return ret;
}

array
cross_a_a( const array& arg1, const array& arg2)
{
	validate_array( arg1);
	validate_array( arg2);
	std::vector<npy_intp> dims1 = shape(arg1);
	std::vector<npy_intp> dims2 = shape(arg2);
	if (dims1 != dims2) {
		throw std::invalid_argument( "Array shape mismatch.");
	}

	array ret = makeNum( dims1);
	const double* arg1_i = (double*)data(arg1);
	const double* arg2_i = (double*)data(arg2);
	double* ret_i = (double*)data(ret);
	double* const ret_stop = ret_i + 3*dims1[0];
	for ( ; ret_i < ret_stop; ret_i += 3, arg1_i += 3, arg2_i += 3) {
		vector ret = vector(arg1_i).cross( vector( arg2_i));
		ret_i[0] = ret.get_x();
		ret_i[1] = ret.get_y();
		ret_i[2] = ret.get_z();
	}
	return ret;
}

array
cross_a_v( const array& arg1, const vector& arg2)
{
	validate_array( arg1);
	std::vector<npy_intp> dims = shape( arg1);
	array ret = makeNum( dims);
	const double* arg1_i = (double*)data( arg1);
	double* ret_i = (double*)data( ret);
	double* const ret_stop = ret_i + 3*dims[0];
	for ( ; ret_i < ret_stop; ret_i += 3, arg1_i += 3) {
		vector ret = vector( arg1_i).cross( arg2);
		ret_i[0] = ret.get_x();
		ret_i[1] = ret.get_y();
		ret_i[2] = ret.get_z();

	}
	return ret;
}

array
cross_v_a( const vector& arg1, const array& arg2)
{
	validate_array( arg2);
	std::vector<npy_intp> dims = shape( arg2);
	array ret = makeNum( dims);
	const double* arg2_i = (double*)data( arg2);
	double* ret_i = (double*)data( ret);
	double* const ret_stop = ret_i + 3*dims[0];
	for ( ; ret_i < ret_stop; ret_i += 3, arg2_i += 3) {
		vector ret = arg1.cross( vector( arg2_i));
		ret_i[0] = ret.get_x();
		ret_i[1] = ret.get_y();
		ret_i[2] = ret.get_z();

	}
	return ret;

}




namespace {
using namespace boost::python;
BOOST_PYTHON_FUNCTION_OVERLOADS( free_rotate, rotate, 2, 3 )
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS( vector_rotate, vector::rotate, 1, 2)
} // !namespace anonymous

struct vector_from_seq
{
	vector_from_seq()
	{
		py::converter::registry::push_back(
			&convertible,
			&construct,
			py::type_id<vector>());
	}

	static void* convertible( PyObject* obj)
	{
		using py::handle;
		using py::allow_null;

		object o( handle<>( borrowed(obj) ) );

		int obj_size = PyObject_Length(obj);
		if (obj_size < 0) {
			PyErr_Clear();
			return 0;
		}
		if (obj_size != 2 && obj_size != 3)
			return 0;
		for(int i=0; i<obj_size; i++)
			if (!py::extract<double>(o[i]).check())
				return 0;
		return obj;
	}

	static void construct(
		PyObject* _obj,
		py::converter::rvalue_from_python_stage1_data* data)
	{
		using namespace boost::python;

		object obj = object(handle<>(borrowed(_obj)));
		void* storage = (
			(boost::python::converter::rvalue_from_python_storage<vector>*)
			data)->storage.bytes;
		int obj_size = PyObject_Length(_obj);
		switch (obj_size) {
			case 1:
				new (storage) vector( py::extract<double>(obj[0]));
				break;
			case 2:
				new (storage) vector(
					py::extract<double>( obj[0]),
					py::extract<double>( obj[1]));
					break;
			case 3: default:
				// Will probably trigger an exception if it is the default
				// case.
				new (storage) vector(
					py::extract<double>( obj[0]),
					py::extract<double>( obj[1]),
					py::extract<double>( obj[2]));
		}
		data->convertible = storage;
	}
};

py::tuple
vector_as_tuple( const vector& v)
{
	return py::make_tuple( v.x, v.y, v.z);
}

vector
vector_pos( const vector& v)
{
	return v;
}


void
wrap_vector()
{
	// Numeric versions for some of the above
	// TODO: round out the set.
	def( "mag", mag_a);
	def( "dot", dot_a);
	def( "cross", cross_a_a);
	def( "cross", cross_a_v);
	def( "cross", cross_v_a);
	def( "mag2", mag2_a);
	def( "norm", norm_a);

	// Free functions for vectors
	// The following two functions have never been implemented:
	//py::def( "det3",a_dot_b_cross_c, "The determinant of the matrix of 3 vectors.");
	//py::def( "cross3",a_cross_b_cross_c, "The vector triple product.");
	py::def( "dot", dot, "The dot product between two vectors.");
	py::def( "cross", cross, "The cross product between two vectors.");
	py::def( "mag", mag, "The magnitude of a vector.");
	py::def( "mag2", mag2, "A vector's magnitude squared.");
	py::def( "norm", norm, "Returns the unit vector of its argument.");
	py::def( "comp", comp, "The scalar projection of arg1 to arg2.");
	py::def( "proj", proj, "The vector projection of arg1 to arg2.");
	py::def( "diff_angle", diff_angle, "The angle between two vectors, in radians.");
	py::def( "rotate", rotate, free_rotate( args("vector", "angle", "axis"),
		"Rotate a vector about an axis vector through an angle.") );

	//AS added throw()

	vector (vector::* truediv)( double) const throw()= &vector::operator/;
	const vector& (vector::* itruediv)( double) throw() = &vector::operator/=;

	// The vector class, constructable from 0, one, two or three doubles.
	py::class_<vector>("vector", py::init< py::optional<double, double, double> >())
		// Explicit copy.
		.def( init<vector>())
		// member variables.
		.add_property( "x", &vector::get_x, &vector::set_x)
		.add_property( "y", &vector::get_y, &vector::set_y)
		.add_property( "z", &vector::get_z, &vector::set_z)
		// Member functions masquerading as properties.
		.add_property( "mag", &vector::mag, &vector::set_mag)
		.add_property( "mag2", &vector::mag2, &vector::set_mag2)
		// Member functions
		.def( "dot", &vector::dot, "The dot product of this vector and another.")
		.def( "cross", &vector::cross, "The cross product of this vector and another.")
		.def( "norm", &vector::norm, "The unit vector of this vector.")
		.def( "comp", &vector::comp, "The scalar projection of this vector onto another.")
		.def( "proj", &vector::proj, "The vector projection of this vector onto another.")
		.def( "diff_angle", &vector::diff_angle, "The angle between this vector "
			"and another, in radians.")
		.def( "clear", &vector::clear, "Zero the state of this vector.  Potentially "
			"useful for reusing a temporary variable.")
		.def( "rotate", &vector::rotate, vector_rotate( "Rotate this vector about "
			"the specified axis through the specified angle, in radians",
			args( "angle", "axis")))
		.def( "__abs__", &vector::mag, "Return the magnitude of this vector.")
		.def( "__pos__", vector_pos, "Return an unmodified copy of this vector.")
		// Some support for the sequence protocol.
		.def( "__len__", &vector::py_len)
		.def( "__getitem__", &vector::py_getitem)
		.def( "__setitem__", &vector::py_setitem)
		// Use this to quickly convert vector's to tuples.
		.def( "astuple", vector_as_tuple, "Convert this vector to a tuple.  "
			"Same as tuple(vector), but much faster.")
		// Member operators
		.def( -self)
		.def( self + self)
		.def( self += self)
		.def( self - self)
		.def( self -= self)
		.def( self * double())
		.def( self *= double())
		.def( self / double())
		.def( self /= double())
		.def( double() * self)
		.def( self == self )
		.def( self != self )

		// This doesn't work either (NPY_FLOAT not recognized as a type):
		//.def( other<NPY_FLOAT>() * self)

		// Suggestion from Jonathan Brandmeyer, which doesn't compile:
		//.def( "__mul__", &vector::operator*(double), "Multiply vector times scalar")
		//.def( "__rmul__", &operator*(const double&, const vector&), "Multiply scalar times vector")

		// Same as self / double, when "from __future__ import division" is in effect.
		.def( "__itruediv__", itruediv, return_value_policy<copy_const_reference>())
		// Same as self /= double, when "from __future__ import division" is in effect.
		.def( "__truediv__",  truediv)
    	.def( self_ns::str(self))        // Support ">>> print foo"
		.def( "__repr__", &vector::repr) // Support ">>> foo"
		;

	// Pass a sequence to some functions that expect type visual::vector.
	vector_from_seq();
}

} // !namespace cvisual
