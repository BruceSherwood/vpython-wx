// This alternative to numpy is not currently being used
// but is retained in CVS for possible future use.
// In particular, information about array objects such
// as curve cannot currently be cached because we don't
// know when a numpy pos array has been changed.

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.
#include "python/scalar_array.hpp"
#include "python/vector_array.hpp"
#include "util/vector.hpp"

#include <boost/python/class.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/init.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/iterator.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL visual_PyArrayHandle
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

namespace cvisual { namespace python {

/***********************  scalar_array implementation  ***********************/

scalar_array::scalar_array( const boost::python::list& sequence)
	: data( boost::python::extract<int>( sequence.attr("__len__")()))
{
	int s_i = 0;
	for ( iterator i = data.begin(); i != data.end(); ++i, ++s_i) {
		*i = boost::python::extract<double>( sequence[s_i]);
	}
}

scalar_array::scalar_array( const boost::python::numeric::array& sequence)
	: data( ((PyArrayObject*)sequence.ptr())->dimensions[0])
{
	const PyArrayObject* seq_ptr = (PyArrayObject*)sequence.ptr();
	if (!( seq_ptr->nd == 1
		&& seq_ptr->descr->type_num == PyArray_DOUBLE)) {
		throw std::invalid_argument( "Must construct a scalar_array from a "
			"one-dimensional array of type Float64");
	}

	const double* seq_i = (const double*)seq_ptr->data;
	iterator i = this->begin();
	for ( ; i != this->end(); ++i, ++seq_i) {
		*i = *seq_i;
	}
}

// Convert to a Numeric.array
boost::python::handle<PyObject>
scalar_array::as_array() const
{
	int dims[] = { this->size() };
	boost::python::handle<> ret( PyArray_FromDims( 1, dims, PyArray_DOUBLE));
	PyArrayObject* ret_ptr = (PyArrayObject*)ret.get();

	double* r_i = (double*)ret_ptr->data;
	const_iterator i = this->begin();
	for ( ; i != this->end(); ++i, ++r_i) {
		*r_i = *i;
	}
	return ret;
}

void
scalar_array::append( double s)
{
	data.push_back( s);
}

void
scalar_array::prepend( double s)
{
	data.push_front( s);
}


void
scalar_array::head_clip()
{
	data.pop_front();
}

void
scalar_array::head_crop( int i_)
{
	if (i_ < 0)
		throw std::invalid_argument( "Cannot crop a negative amount.");
	size_t i = (size_t)i_;
	if (i >= data.size())
		throw std::out_of_range( "Cannot crop greater than the array's length.");

	iterator begin = data.begin();
	data.erase( begin, begin+i);
}

void
scalar_array::tail_clip()
{
	data.pop_back();
}

void
scalar_array::tail_crop( int i_)
{
	if (i_ < 0)
		throw std::invalid_argument( "Cannot crop a negative amount.");
	size_t i = (size_t)i_;
	if (i >= data.size())
		throw std::out_of_range( "Cannot crop greater than the array's length.");

	iterator end = data.end();
	data.erase( end-i, end);
}

scalar_array
scalar_array::operator*( double s) const
{
	scalar_array ret( data.size());
	iterator r_i = ret.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
		*r_i = *i * s;
	}
	return ret;
}


scalar_array
scalar_array::operator*( const scalar_array& s) const
{
	if (data.size() != s.data.size())
		throw std::out_of_range( "Incompatible array multiplication.");

	scalar_array ret( data.size());

	const_iterator s_i = s.begin();
	iterator r_i = ret.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++s_i, ++r_i) {
		*r_i = *i * *s_i;
	}
	return ret;
}

vector_array
scalar_array::operator*( const vector_array& v) const
{
	if (data.size() != v.data.size())
		throw std::out_of_range( "Incompatible array multiplication.");

	vector_array ret( data.size());

	vector_array::iterator r_i = ret.begin();
	vector_array::const_iterator v_i = v.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i, ++v_i) {
		*r_i = *i * *v_i;
	}
	return ret;
}

vector_array
scalar_array::operator*( const vector& v) const
{
	vector_array ret( data.size());

	vector_array::iterator r_i = ret.begin();
	for (const_iterator i = data.begin(); i!= data.end(); ++i) {
		*r_i = v * *i;
	}
	return ret;
}

const scalar_array&
scalar_array::operator*=( double s)
{
	for (iterator i = data.begin(); i != data.end(); ++i) {
		*i *= s;
	}
	return *this;
}

const scalar_array&
scalar_array::operator*=( const scalar_array& s)
{
	if (data.size() != s.data.size())
		throw std::out_of_range( "Incompatible array multiplication.");

	const_iterator s_i = s.begin();
	for (iterator i = data.begin(); i != data.end(); ++i, ++s_i) {
		*i *= *s_i;
	}
	return *this;
}

scalar_array
scalar_array::operator/( double s) const
{
	scalar_array ret( data.size());
	iterator r_i = ret.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
		*r_i = *i / s;
	}
	return ret;
}

scalar_array
scalar_array::operator/( const scalar_array& s) const
{
	if (data.size() != s.data.size())
		throw std::out_of_range( "Incompatible array division.");

	scalar_array ret( data.size());

	const_iterator s_i = s.begin();
	iterator r_i = ret.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++s_i, ++r_i) {
		*r_i = *i / *s_i;
	}
	return ret;
}

const scalar_array&
scalar_array::operator/=( double s)
{
	for (iterator i = data.begin(); i != data.end(); ++i) {
		*i /= s;
	}
	return *this;
}

const scalar_array&
scalar_array::operator/=( const scalar_array& s)
{
	if (data.size() != s.data.size())
		throw std::out_of_range( "Incompatible array division.");

	const_iterator s_i = s.begin();
	for (iterator i = data.begin(); i != data.end(); ++i, ++s_i) {
		*i *= *s_i;
	}
	return *this;
}

scalar_array
scalar_array::operator+( const scalar_array& s) const
{
	if (data.size() != s.data.size())
		throw std::out_of_range( "Incompatible array addition.");

	scalar_array ret( data.size());

	iterator r_i = ret.begin();
	const_iterator s_i = s.data.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++s_i, ++r_i) {
		*r_i = *s_i + *i;
	}
	return *this;
}


scalar_array
scalar_array::operator+( double s) const
{
	scalar_array ret( data.size());
	iterator r_i = ret.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
		*r_i = *i + s;
	}
	return ret;
}

const scalar_array&
scalar_array::operator+=( double s)
{
	for (iterator i = data.begin(); i != data.end(); ++i) {
		*i += s;
	}
	return *this;
}


const scalar_array&
scalar_array::operator+=( const scalar_array& s)
{
	if (data.size() != s.data.size())
		throw std::out_of_range( "Incompatible array addition.");

	const_iterator s_i = s.begin();
	for (iterator i = data.begin(); i != data.end(); ++i, ++s_i) {
		*i += *s_i;
	}
	return *this;
}


scalar_array
scalar_array::operator-( const scalar_array& s) const
{
	if (data.size() != s.data.size())
		throw std::out_of_range( "Incompatible array subtraction.");

	scalar_array ret( data.size());

	iterator r_i = ret.begin();
	const_iterator s_i = s.data.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++s_i, ++r_i) {
		*r_i = *i - *s_i;
	}
	return ret;
}

scalar_array
scalar_array::operator-( double s) const
{
	scalar_array ret( data.size());
	iterator r_i = ret.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
		*r_i = *i - s;
	}
	return ret;
}


const scalar_array&
scalar_array::operator-=( double s)
{
	for (iterator i = data.begin(); i != data.end(); ++i) {
		*i -= s;
	}
	return *this;
}


const scalar_array&
scalar_array::operator-=( const scalar_array& s)
{
	if (data.size() != s.data.size())
		throw std::out_of_range( "Incompatible array subtraction.");

	const_iterator s_i = s.begin();
	for (iterator i = data.begin(); i != data.end(); ++i, ++s_i) {
		*i -= *s_i;
	}
	return *this;
}


scalar_array
scalar_array::operator-() const
{
	scalar_array ret( data.size());
	iterator r_i = ret.begin();
	for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
		*r_i = - *i;
	}
	return ret;
}

void
scalar_array::py_setitem( int index, double value)
{
	if (index < 0)
		// Negative indexes are counted from the end of the array.
		index += data.size();

	data.at(index) = value;
}

double
scalar_array::py_getitem( int index)
{
	if (index < 0)
		index += data.size();

	return data.at(index);
}


double
scalar_array::sum() const
{
	double ret = 0.0;
	for ( const_iterator i = data.begin(); i != data.end(); ++i) {
		ret += *i;
	}
	return ret;
}

void
wrap_scalar_array()
{
	using namespace boost::python;

	scalar_array (scalar_array::* truediv_self)( const scalar_array&) const = &scalar_array::operator/;
	const scalar_array& (scalar_array::* itruediv_self)( const scalar_array&) = &scalar_array::operator/=;
	scalar_array (scalar_array::* truediv_double)( double) const = &scalar_array::operator/;
	const scalar_array& (scalar_array::* itruediv_double)( double) = &scalar_array::operator/=;

	class_<scalar_array>( "scalar_array", init< optional<int, double> >( args( "size", "fill" ) ))
		.def( init<const list&>())
		.def( init<numeric::array>())
		.def( self + self)
		.def( self + other<double>())
		.def( self += self)
		.def( self += other<double>())
		.def( self - self)
		.def( self - other<double>())
		.def( self -= self)
		.def( self -= other<double>())
		.def( self * self)
		.def( self * other<double>())
		.def( self * other<vector>())
		.def( self *= self)
		.def( self *= other<double>())
		.def( self * other<vector_array>())
		.def( self / self)
		.def( self / other<double>())
		.def( self /= self)
		.def( self /= other<double>())      // in-place division by single element.
		.def( "__truediv__", truediv_self)
		.def( "__truediv__", truediv_double)
		.def( "__itruediv__", itruediv_double, return_value_policy<copy_const_reference>())
		.def( "__itruediv__", itruediv_self, return_value_policy<copy_const_reference>())
		.def( -self)
		.def( "__iter__", iterator<scalar_array>())
		.def( "__len__", &scalar_array::size)
		.def( "__getitem__", &scalar_array::py_getitem)
		.def( "__setitem__", &scalar_array::py_setitem)
		.def( "append", &scalar_array::append)
		.def( "prepend", &scalar_array::prepend)
		.def( "head_clip", &scalar_array::head_clip)
		.def( "head_crop", &scalar_array::head_crop)
		.def( "tail_clip", &scalar_array::tail_clip)
		.def( "tail_crop", &scalar_array::tail_crop)
		.def( "sum", &scalar_array::sum, "Returns the sum of all elements in the array.")
		.def( "as_array", &scalar_array::as_array,
			"Returns a new self.__len__() x 1 Numeric.array from this scalar_array.")
		;

}

} } // !namespace cvisual::python
