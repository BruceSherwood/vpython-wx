#ifndef VPYTHON_UTIL_TMATRIX_HPP
#define VPYTHON_UTIL_TMATRIX_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/vector.hpp"
#include <cstring>

namespace cvisual {

struct vertex
{
	double x;
	double y;
	double z;
	double w;

	inline vertex( double _x=0, double _y=0, double _z=0, double _w=1.0)
		: x(_x), y(_y), z(_z), w(_w) {}

	explicit inline vertex( const vector& v, double _w = 1.0)
		: x( v.x), y(v.y), z(v.z), w(_w) {}

	vector project() const
	{ double w_i = 1.0/w; return vector( x*w_i, y*w_i, z*w_i); }

	inline void
	gl_render() const
	{ glVertex4d( x, y, z, w); }

	double& operator[] (int i) { return (&x)[i]; }
	double operator[] (int i) const { return (&x)[i]; }
};

/** A double-precision 3D affine transformation matrix. */
class tmatrix
{
private:
	/** This is a double-precision matrix in _COLUMN MAJOR ORDER_.  User's beware.
	  It is in this order since that is what OpenGL uses internally - thus
	  eliminating a reformatting penalty.
	*/
	double M[4][4];

    friend void inverse( tmatrix& ret, const tmatrix& arg);
public:
	/** Returns the address of the first element in the matrix.  Ideally, this
	 * function should not exist.
	 */
    inline const double* matrix_addr() const { return M[0]; }

	/** Create a new tmatrix, initialized to the identity matrix. */
	inline tmatrix() throw() { ident(); }

	/** Make a deep copy of t. */
	inline tmatrix( const tmatrix& t ) throw()
	{ std::memcpy(M, t.M, sizeof(M)); }

	/** Initialize this matrix to A * B */
	inline tmatrix( const tmatrix& A, const tmatrix& B ) throw()
	{ *this = A * B; }

	/** Returns the identity matrix. */
	static const tmatrix& identity() throw();

	/** Sets this matrix to the identity and returns an rvalue reference to self. */
	inline const tmatrix&
	ident( void) throw()
	{
		x_column();
		y_column();
		z_column();
		w_column();
		w_row();
		return *this;
	}

	/** Address an individual element of the tmatrix.  The internal format of
	  the matrix may be anything, so use this function to reliably get the
	  individual elements.
	*/
	inline const double&
	operator()( size_t row, size_t column) const
	{ return M[column][row]; }

	/** Address an individual element of the tmatrix.  The internal format of
	  the matrix may be anything, so use this function to reliably get the
	  individual elements.
	*/
	inline double&
	operator()( size_t row, size_t column)
	{ return M[column][row]; }

	/** Sets the first column to v */
	inline void x_column( const vector& v) throw()
	{
		M[0][0] = v.x;
		M[0][1] = v.y;
		M[0][2] = v.z;
	}

	/** Sets the second column to v */
	inline void y_column( const vector& v) throw()
	{
		M[1][0] = v.x;
		M[1][1] = v.y;
		M[1][2] = v.z;
	}

	/** Sets the third column to v */
	inline void z_column( const vector& v) throw()
	{
		M[2][0] = v.x;
		M[2][1] = v.y;
		M[2][2] = v.z;
	}

	/** Sets the fourth column to v */
	inline void w_column( const vector& v) throw()
	{
		M[3][0] = v.x;
		M[3][1] = v.y;
		M[3][2] = v.z;
	}

	/** Sets the first column to x, y, z */
	inline void x_column( double x=1, double y=0, double z=0) throw()
	{
		M[0][0] = x;
		M[0][1] = y;
		M[0][2] = z;
	}

	/** Sets the second column to x, y, z */
	inline void y_column( double x=0, double y=1, double z=0) throw()
	{
		M[1][0] = x;
		M[1][1] = y;
		M[1][2] = z;
	}

	/** Sets the third column to x, y, z */
	inline void z_column( double x=0, double y=0, double z=1) throw()
	{
		M[2][0] = x;
		M[2][1] = y;
		M[2][2] = z;
    }

	/** Sets the fourth column to x, y, z */
	inline void w_column(double x=0, double y=0, double z=0) throw()
	{
		M[3][0] = x;
		M[3][1] = y;
		M[3][2] = z;
	}

	/** Sets the bottom row to x, y, z, w */
	inline void w_row(double x=0, double y=0, double z=0, double w=1) throw()
	{
		M[0][3]=x;
		M[1][3]=y;
		M[2][3]=z;
		M[3][3]=w;
	}

	/** Projects v using the current tmatrix values. */
	vertex project(const vector& v) const throw();

	/** An alias for operator*= */
	void concat(const tmatrix& A, const tmatrix& B) throw();

	// Right-multiply this matrix by a scaling matrix.
	void scale( const vector& v, const double w = 1);

	// Right multiply the matrix by a translation matrix
	void translate( const vector& v );

	/** Postcondition: *this == *this * other */
	const tmatrix&
	operator*=( const tmatrix& other);

	/** Multiply this matrix by another one. */
	tmatrix
	operator*( const tmatrix& other) const;

	void invert_ortho(const tmatrix& A) throw();

	/** M^-1 * [x y z w] */
	vector times_inv( const vector& v, double w = 1.0) const throw();

	/** multiplication by a vector [x y z 0] */
	vector times_v( const vector& v) const throw();

	/** multiplication by a point [x y z 1] */
	vector operator*( const vector& v) const throw();

	/** multiplication by an arbirary vertex [x y z w] */
	vertex operator*( const vertex& v) const throw();

	/** multiplication by [0 0 0 1] **/
	vector origin() const throw();

	/** Overwrites the currently active matrix in OpenGL with this one. */
	inline void
	gl_load(void) const
	{ glLoadMatrixd( M[0]); }

	/** Multiplies the active OpenGL matrix by this one. */
	inline void
	gl_mult(void) const
	{ glMultMatrixd( M[0]); }

	/** Initialize this tmatrix with the contents of the OpenGL modelview,
	  * texture, color, or projection matricies.
	  * @return *this.
	  */
	tmatrix& gl_modelview_get();
	tmatrix& gl_texture_get();
	tmatrix& gl_color_get();
	tmatrix& gl_projection_get();
	/**
	 * Dump this matrix to a formatted string.
	 */
	std::string to_string() const;
};

// Compute the inverse of arg, and store it in ret.
void inverse( tmatrix& ret, const tmatrix& arg);

// Returns a rotation matrix to perform rotations about an axis passing through
// the origin through an angle in the direction specified by the Right Hand Rule.
tmatrix rotation( double angle, const vector& axis);
// Returns a rotation matrix to perform rotations about an axis passing through
// origin in the direction axis as specified by the Right Hand Rule.
tmatrix rotation( double angle, const vector& axis, const vector& origin);

// Pushes its constructor argument onto the active OpenGL matrix stack, and
// multiplies the active matrix by the new one when constructed, and pops it off
// when destructed.
class gl_matrix_stackguard
{
 private:
	// Avoid calls that are nonsensical for this class.
	gl_matrix_stackguard( const gl_matrix_stackguard&);
	const gl_matrix_stackguard&
	operator=( const gl_matrix_stackguard&);

 public:
	// A stackguard that only performs a push onto the matrix stack.
	// Postcondition: the stack is one matrix taller, but identical to before.
	inline gl_matrix_stackguard() { glPushMatrix(); }
	gl_matrix_stackguard( const tmatrix&);
	inline ~gl_matrix_stackguard() { glPopMatrix(); }
};

} // !namespace cvisual

#endif // !VPYTHON_UTIL_TMATRIX_HPP
