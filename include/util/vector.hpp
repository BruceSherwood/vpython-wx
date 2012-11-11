#ifndef VPYTHON_UTIL_VECTOR_HPP
#define VPYTHON_UTIL_VECTOR_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "wrap_gl.hpp"
#include <boost/python/numeric.hpp>
#include <iosfwd>
#include <cmath>
#include <cassert>
#include <sstream>

namespace cvisual {

class vector
{
public:
	double x;
	double y;
	double z;

public:
	explicit vector( double a = 0.0, double b = 0.0, double c = 0.0) throw()
		: x(a), y(b), z(c) {}

	inline explicit vector( const double* v)
		: x(v[0]), y(v[1]), z(v[2]) {}

	// Overloaded binary +, -, *, and /
	inline vector
	operator+( const vector& v) const throw()
	{ return vector( x+v.x, y+v.y, z+v.z); }

	inline vector
	operator-( const vector& v) const throw()
	{ return vector( x-v.x, y-v.y, z-v.z); }

	inline vector
	operator*( const double s) const throw()
	{ return vector( s*x, s*y, s*z); }

	// Element-wise multiplication used in frame.cpp; not exposed to users
	inline vector
	operator*( const vector& v) const throw()
	{ return vector( x*v.x, y*v.y, z*v.z); }

	inline vector
	operator/( const double s) const throw()
	{ return vector( x/s, y/s, z/s); }

    // This operator describes a strict weak ordering as defined by the STL.
	bool
	stl_cmp( const vector& v) const;

	inline bool
	operator==( const vector& v) const throw()
	{ return (v.x == this->x && v.y == this->y && v.z == this->z); }

	inline bool
	operator!=( const vector& v) const throw()
	{ return !(v == *this); }

	// Overloaded uniary !, probably bad coding practice.
	inline bool
	operator!( void) const throw()
	{ return !x && !y && !z; }

	bool nonzero() const throw() { return x || y || z; }

    // Overloaded assignment: +=, -=, *=, /=
	inline const vector&
	operator+=( const vector& v) throw()
	{ x=x+v.x; y=y+v.y; z=z+v.z; return *this; }

	inline const vector&
	operator-=( const vector& v) throw()
	{ x=x-v.x; y=y-v.y; z=z-v.z; return *this; }

	inline const vector&
	operator*=( const double s) throw()
	{ x=x*s; y=y*s; z=z*s; return *this; }

	inline const vector&
	operator/=( const double s) throw()
	{ x=x/s; y=y/s; z=z/s; return *this; }

 	inline vector
	operator-() const throw()
	{ return vector( -x, -y, -z); }

	// return the magnitude of this vector
	inline double
	mag( void) const throw()
	{ return std::sqrt( x*x + y*y + z*z); }

	// This is a magnitude algorithm that is intended to be stable at values
	// greater than 1e154 (or so).  It is much slower since it uses sin, cos,
	// and atan to get the result.
	double
	stable_mag(void) const;

	// return the square of the this vector's magnitude
	inline double
	mag2( void) const throw()
	{ return (x*x + y*y + z*z); }

	// return the unit vector of this vector
	vector
	norm( void) const throw();

	inline void
	set_mag( double m) throw()
	{ *this = norm()*m; }

	inline void
	set_mag2( double m2) throw()
	{ *this = norm()*std::sqrt(m2); }
	// Pythonic function to provide a "representation" of this object.
	// object.__repr__() should return a string that, were it executed as python
	// code, should regenerate the object.
	std::string
	repr() const;

	// return the dot product of this vector and another
	inline double
	dot( const vector& v) const throw()
	{ return ( v.x * this->x + v.y * this->y + v.z * this->z); }

	// Return the cross product of this vector and another.
	vector
	cross( const vector& v) const throw();

	// Return the scalar triple product
	double
	dot_b_cross_c( const vector& b, const vector& c) const throw();

	// Return the vector triple product
	vector
	cross_b_cross_c( const vector& b, const vector& c) const throw();

	// Scalar projection of this to v
	double
	comp( const vector& v) const throw();

	// Vector projection of this to v
	vector
	proj( const vector& v) const throw();

	// Returns the angular difference between two vectors, in radians, between 0 and pi.
	double
	diff_angle( const vector& v) const throw();

	// Scale this vector to another, by elementwise multiplication
	inline vector
	scale( const vector& v) const throw()
	{ return vector( this->x*v.x, this->y*v.y, this->z*v.z); }

    // Inversely scale this vector to another, by elementwise division
    inline vector
    scale_inv( const vector& v) const throw()
    { return vector( x/v.x, y/v.y, z/v.z); }

	vector
	rotate( double angle, vector axis = vector(0,0,1)) throw();

	// Last ditch direct read/write access to the private variables
	inline double
	get_x( void) const throw() { return x; }

	inline void
	set_x( double s) throw() { this->x = s; }

	inline double
	get_y( void) const throw() { return y; }

	inline void
	set_y( double s) throw() { this->y = s; }

	inline double
	get_z( void) const throw() { return z; }

	inline void
	set_z( double s) throw() { this->z = s; }

	// zero the state of the vector. Potentially useful for reusing a temporary.
	inline void
	clear( void) { x=0.0; y=0.0; z=0.0; }

    inline int
	py_len() { return 3; }

	double py_getitem( int i) const;

	void py_setitem(int i, double value);


	inline double&
	operator[]( size_t ref)
	{
		assert( ref < 3);
		switch (ref) {
			case 0:
				return x;
			case 1:
				return y;
			case 2:
				return z;
			default:
				assert( true == false);
		}
	}

	inline const double&
	operator[]( size_t ref) const
	{
		assert( ref < 3);
		switch (ref) {
			case 0:
				return x;
			case 1:
				return y;
			case 2:
				return z;
		}
	}

	inline vector
	fabs() const
	{ return vector( std::fabs(x), std::fabs(y), std::fabs(z)); }

	inline void
	gl_render() const
	{ glVertex3dv( &x); }

	inline void
	gl_normal() const
	{ glNormal3dv( &x); }

	inline double
	sum() const
	{ return x + y + z; }
};

// Free functions for mag, mag2, dot, unit, cross, and tripleproducts.
// All of these functions merely call their class-member variants to save code.
inline double
mag( const vector& v)
{ return v.mag(); }

inline double
mag2( const vector& v)
{ return v.mag2(); }

inline vector
norm( const vector& v)
{ return v.norm(); }

inline double
dot( const vector& v1, const vector& v2)
{ return v1.dot( v2); }

inline vector
cross( const vector& v1, const vector& v2)
{ return v1.cross( v2); }

inline double
a_dot_b_cross_c( const vector& a, const vector& b, const vector& c)
{  return a.dot_b_cross_c( b, c);  }

inline vector
a_cross_b_cross_c( const vector& a, const vector& b, const vector& c)
{ return a.cross_b_cross_c( b, c); }

// Scalar projection of v1 -> v2
inline double
comp( const vector& v1, const vector& v2)
{ return v1.comp( v2); }

// Vector projection of v1 to v2
inline vector
proj( const vector& v1, const vector& v2)
{ return v1.proj( v2); }

// Returns the angular difference between two vectors, in radians, from 0 - pi.
inline double
diff_angle( const vector& v1, const vector& v2)
{ return v1.diff_angle( v2); }

inline vector
rotate( vector v, double angle, const vector axis = vector( 0,0,1))
{ return v.rotate( angle, axis); }


// Definitions of the global functions for operator *, with a vector on the RHS,
// and scalar on the LHS.

inline vector
operator*( const double& s, const vector& v)
{
  return vector( s*v.x, s*v.y, s*v.z);
}
} // !namespace cvisual

// We should not need to place this in namespace std, but GCC's Koenig L/U fails
//   if we don't.
namespace std {
// Insertion operator.  Example output: <xxxx, yyyy, zzzz>
// Based on "The C++ Standard Library", N. M. Josuttis, section 13.12.1
template<typename char_T, typename traits>
basic_ostream<char_T, traits>&
operator<<( basic_ostream<char_T, traits>& stream, const cvisual::vector& v)
{
	basic_ostringstream<char_T, traits> s;
	s.copyfmt( stream);
	s.width( 0);

	s << "<" << v.x << ", " << v.y << ", " << v.z << ">";
	stream << s.str();

	return stream;
}

} // !namespace std

namespace cvisual {

typedef vector shared_vector;

} // !namespace cvisual

#endif // !VPYTHON_UTIL_VECTOR_HPP
