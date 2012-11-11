#ifndef VPYTHON_PYTHON_SCALAR_ARRAY_HPP
#define VPYTHON_PYTHON_SCALAR_ARRAY_HPP

// This alternative to numpy is not currently being used
// but is retained in CVS for possible future use.
// In particular, information about array objects such
// as curve cannot currently be cached because we don't
// know when a numpy pos array has been changed.

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/vector.hpp"
#include <deque>

#include <boost/python/numeric.hpp>

namespace cvisual { namespace python {

class vector_array;

class scalar_array
{
 private:
	std::deque<double> data;
	friend class vector_array;
	
 public:
	typedef std::deque<double>::iterator iterator;
	typedef std::deque<double>::const_iterator const_iterator;
	
	inline scalar_array( int size = 0, double fill = 0)
		: data( size, fill) {}
	
	// Construct from a continuous 1-D sequence (tuple or list)
	explicit scalar_array( const boost::python::list& sequence);
	explicit scalar_array( const boost::python::numeric::array& sequence);
	
	inline scalar_array( const scalar_array& other)
		: data( other.data) {}
	
	boost::python::handle<PyObject>
	as_array() const;
	
	inline iterator
	begin() { return data.begin(); }
	
	inline const_iterator
	begin() const { return data.begin(); }
	
	inline iterator
	end() { return data.end(); }
	
	inline const_iterator
	end() const { return data.end(); }
		
	// Append a single element to the array.
	void append( double s);
	
	// Prepend a single element the the array.
	void prepend( double s);
	
	// Remove a single element from the beginning of the array
	void head_clip();
	// Remove i elemnts from the beginning of the array.
	void head_crop( int i);
	// Remove a single element from the end of the array.
	void tail_clip();
	// Remove i elements from the end of the array.
	void tail_crop( int i);
	
	scalar_array
	operator*( double s) const;
	
	scalar_array
	operator*( const scalar_array& s) const;
	
	vector_array
	operator*( const vector_array& v) const;

	vector_array
	operator*( const vector& v) const;
	
	const scalar_array&
	operator*=( double s);
	
	const scalar_array&
	operator*=( const scalar_array& s);
	
	scalar_array
	operator/( double s) const;
	
	scalar_array
	operator/( const scalar_array& s) const;
	
	const scalar_array&
	operator/=( double s);
	
	const scalar_array&
	operator/=( const scalar_array& s);
	
	scalar_array
	operator+( const scalar_array& s) const;
	
	scalar_array
	operator+( double s) const;
	
	const scalar_array&
	operator+=( double s);
	
	const scalar_array&
	operator+=( const scalar_array& s);
	
	scalar_array
	operator-( const scalar_array& s) const;
	
	scalar_array
	operator-( double s) const;
	
	const scalar_array&
	operator-=( double s);
	
	const scalar_array&
	operator-=( const scalar_array& s);
	
	scalar_array
	operator-() const;
	
	inline double&
	operator[]( int i) { return data[i]; }
	
	inline const double&
	operator[]( int i) const { return data[i]; }
	
	// Returns the number of elemnts in the array.
	inline int 
	size() const { return data.size(); }
	
	double
	py_getitem( int index);
	
	void
	py_setitem( int index, double value);

	double
	sum() const;

};

void wrap_scalar_array();	
	
} } // !namepace cvisual::python

#endif // !VPYTHON_PYTHON_SCALAR_ARRAY_HPP
