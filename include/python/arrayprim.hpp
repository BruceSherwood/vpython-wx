#ifndef VPYTHON_PYTHON_ARRAYPRIM_H
#define VPYTHON_PYTHON_ARRAYPRIM_H

// Attempt to refactor all the redundant and buggy code in the array primitives.
// Frankly I'm not that happy with this design, but it's better than what was here
//   before.

#include "renderable.hpp"
#include "python/num_util.hpp"
#include "python/slice.hpp"

namespace cvisual { namespace python {

// An Nx3 array of CTYPES, specialized for use in array primitives.  This class
// should not go anywhere except inside an array primitive, not even as a return
// value for primitive.pos or whatever.
template <class CTYPE>
class arrayprim_array : public array, private boost::noncopyable {
protected:
	size_t length;     // number of points in the array primitive
	size_t allocated;  // == shape(*this)[0]

public:
	arrayprim_array();
	arrayprim_array( const arrayprim_array& r )  //< Actually copies, to avoid aliasing between array primitives
		: array(object(r)) {}

	void set_length( size_t new_len );

	CTYPE* data(int index=0) { return (CTYPE*)cvisual::python::data(*this) + index*3; }
	CTYPE* end() { return data(length); }

	const CTYPE* data(int index=0) const { return (const CTYPE*)cvisual::python::data(*this) + index*3; }
	const CTYPE* end() const { return data(length); }
};

class arrayprim : public renderable {
protected:
	size_t count;
	virtual void set_length(size_t);

	slice all() { return slice(0,count); }

	arrayprim_array<double> pos;

public:
	arrayprim();

	boost::python::object get_pos(void);

	void set_pos( const double_array& pos );    // An Nx3 array of doubles
	void set_pos_v( const vector& pos );        // Interpreted as a single point

	void set_x( const double_array& x);
	void set_x_d( const double x );
	void set_y( const double_array& y);
	void set_y_d( const double y );
	void set_z( const double_array& z);
	void set_z_d( const double z );

	void append( const vector& _pos, int retain );
	void append( const vector& _pos ) { append( _pos, -1 ); }
};

class arrayprim_color : public arrayprim {
protected:
	virtual void set_length(size_t);

	arrayprim_array<double> color;

public:
	arrayprim_color();

	boost::python::object get_color(void);

	void set_color( const double_array& color ); // An Nx3 array of color floats
	//void set_color_t( const rgb& color );        // A single tuple - appears to be dealt with fine by implicit conversions?

	void set_red( const double_array& red );
	void set_red_d( const double red );
	void set_blue( const double_array& blue );
	void set_blue_d( const double blue );
	void set_green( const double_array& green );
	void set_green_d( const double green );

	using arrayprim::append;
	void append_rgb( const vector& _pos, double red=-1, double green=-1, double blue=-1, int retain=-1 );
	void append( const vector& _pos, const rgb& _color, int retain ); // Append a single position with new color.
	void append( const vector& _pos, const rgb& _color ) { append( _pos, _color, -1 ); }
};

} } // namespace cvisual::python

#endif
