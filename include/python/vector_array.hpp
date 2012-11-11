#ifndef VPYTON_PYTHON_VECTOR_ARRAY_HPP
#define VPYTHON_PYTHON_VECTOR_ARRAY_HPP

// This alternative to numpy is not currently being used
// but is retained in CVS for possible future use.
// In particular, information about array objects such
// as curve cannot currently be cached because we don't
// know when a numpy pos array has been changed.

// Copyright (c) 2003, 2004 Jonathan Brandmeyer.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include <deque>
#include "util/vector.hpp"

#include <boost/python/numeric.hpp>

namespace cvisual { namespace python {
	
class scalar_array;

class vector_array
{
 private:
	std::deque<vector> data;
	friend class scalar_array;
	
 public:
	typedef std::deque<vector>::iterator iterator;
	typedef std::deque<vector>::const_iterator const_iterator;
	
	vector_array( int size = 0, vector fill = vector())
		: data( size, fill){}
	
	// Construct from a list of three-element tuples.
	explicit vector_array( const boost::python::list&);
	vector_array( const vector_array& v)
		: data( v.data) {}
	
	explicit vector_array( boost::python::numeric::array);

	inline iterator
	begin()
	{ return data.begin(); }
	
	inline const_iterator
	begin() const
	{ return data.begin(); }
	
	inline iterator
	end()
	{ return data.end(); }
	
	inline const_iterator
	end() const
	{ return data.end(); }
	
	
	// Append a single vector (or vector represented as a tuple) to the array.
	void append( const vector& v);
	void append( const vector_array& va);
	
	// Prepend a single vector (or vector represented as a tuple) the the array.
	void prepend( const vector& v);
	
	// Remove a single element from the beginning of the array
	void head_clip();
	// Remove i elemnts from the beginning of the array.
	void head_crop( int i);
	// Remove a single element from the end of the array.
	void tail_clip();
	// Remove i elements from the end of the array.
	void tail_crop( int i);
	
	// Scalar operations
	vector_array
	operator*( double s) const;
	
	vector_array
	operator*( const scalar_array& s) const;

	vector_array
	operator*( vector s) const;

	vector_array
	operator/( double s) const;
	
	vector_array
	operator/( const scalar_array& s) const;
	
	vector_array
	operator-() const;
	
	const vector_array&
	operator*=( double s);
	
	const vector_array&
	operator*=( const scalar_array& s);
	
	const vector_array&
	operator/=( double s);
	
	const vector_array&
	operator/=( const scalar_array& s);
	
	
	// Vector operations
	vector_array
	operator+( const vector& v) const;
	
	vector_array
	operator-( const vector& v) const;
	
	vector_array
	operator+( const vector_array& v) const;
	
	vector_array
	operator-( const vector_array& v) const;
	
	const vector_array& 
	operator+=( const vector& v);
	
	const vector_array&
	operator-=( const vector& v);
	
	const vector_array&
	operator+=( const vector_array& v);
	
	const vector_array&
	operator-=( const vector_array& v);
	
	// Compounded operations
	vector_array
	cross( const vector& v);
	
	vector_array
	cross( const vector_array& v);
	
	vector_array
	norm() const;
	
	scalar_array
	dot( const vector& v);
	
	scalar_array
	dot( const vector_array& v);
	
	vector_array
	proj( const vector_array& v);
	
	vector_array
	proj( const vector& v);
	
	void
	rotate( const double& angle, vector axis = vector(0,0,0));
	
	scalar_array
	mag() const;
	
	scalar_array
	mag2() const;
	
	scalar_array
	comp( const vector& v);
	
	scalar_array
	comp( const vector_array& v);
	
	// Returns the vector at the specified position.
	// Use the non-checked version for C++, and the checked version for python.
	inline vector&
	operator[]( int i) { return data[i]; }
	
	inline const vector&
	operator[]( int i) const { return data[i]; }
	
	// Returns the number of elemnts in the array.
	inline int 
	size() const { return data.size(); }
	
	inline bool
	empty() const { return data.empty(); }
	
	vector&
	py_getitem( int index);
	
	void
	py_setitem( int index, vector value);

	scalar_array get_x() const;
	scalar_array get_y() const;
	scalar_array get_z() const;
	
	void set_x( const scalar_array&);
	void set_y( const scalar_array&);
	void set_z( const scalar_array&);
	
	void set_x( boost::python::numeric::array);
	void set_y( boost::python::numeric::array);
	void set_z( boost::python::numeric::array);
	
	// Force every element to be this single value.
	void set_x( double x);
	void set_y( double y);
	void set_z( double z);

	void set_x( const boost::python::list&);
	void set_y( const boost::python::list&);
	void set_z( const boost::python::list&);

	vector
	sum() const;

	// Support expressions matching vector_array = vector - vector_array.
	vector_array
	lhs_sub( const vector& v) const;

	// Array relational comparisons.  These operators return a vector_array whose x,y,and z members
	// are all either 0.0 or 1.1.
	vector_array
	operator>=( const double&) const;

	vector_array
	operator>=( const scalar_array&) const;

	vector_array
	operator>=( const vector_array&) const;

	vector_array
	operator<=( const double&) const;

	vector_array
	operator<=( const scalar_array&) const;

	vector_array
	operator<=( const vector_array&) const;

	vector_array
	fabs() const;

	// Element-wise multiplication...
	vector_array
	operator*( const vector_array& v) const;
	
	boost::python::handle<PyObject>
	as_array() const;
};

void wrap_vector_array();

// Return an ordered list of collisions detected between a set of spheres
// whose centers are encoded in pos, and radii are encoded in radius.
// This algorithm runs in O(n*n) time.
// The returned list is an ordered list of tuple indexes into the array pos.
// e.g., [(2,4)] indicates a collision between the second and fourth spheres.
boost::python::list
sphere_collisions( const vector_array& pos, const scalar_array& radius);

boost::python::list
sphere_to_plane_collisions( const vector_array& pos, const scalar_array& radius
                          , vector normal, vector origin);


// Support several expressions with a vector_array on the rhs
// Note that all of the scalar_array op vector_array expressions are covered
// in class scalar_array.
inline vector_array
operator*( double s, const vector_array& v)
{
	return v * s;
}

inline vector_array
operator-( const vector& v, const vector_array& v_a)
{
	return v_a.lhs_sub( v);
}

inline vector_array
operator+( const vector& v, const vector_array& v_a)
{
	return v_a + v;
}

} } // !namespace cvisual::python

#endif // !VISUAL_VECTOR_ARRAY_H
