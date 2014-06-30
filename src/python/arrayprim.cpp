#include "python/arrayprim.hpp"
#include "python/slice.hpp"

namespace cvisual { namespace python {

using boost::python::object;
using boost::python::make_tuple;
using boost::python::tuple;

template <class CTYPE>
arrayprim_array<CTYPE>::arrayprim_array()
 : array(NULL), length(0), allocated(256)
{
	std::vector<npy_intp> dims(2);
	dims[0] = allocated;
	dims[1] = 3;
	array::operator=( makeNum( dims, (NPY_TYPES)type_npy_traits<CTYPE>::npy_type ) );
}

template <class CTYPE>
void arrayprim_array<CTYPE>::set_length( size_t new_len ) {
	using cvisual::python::slice;

	size_t old_len = length;

	if (new_len == old_len) { return; } // no need to adjust the length

	if (new_len < old_len ) {
		// Shrink, keeping the last points (for retain)
		//(*this)[ slice(0,new_len) ] = (*this)[ slice(old_len-new_len,old_len) ];
		memmove( data(0), data(old_len-new_len), sizeof(CTYPE) * new_len * 3 );
	}
	if (!old_len && allocated) old_len = 1;  // The very first point is meaningful even when length is 0; that's how an empty curve can have a color

	if (new_len > allocated) {
		// Expand allocated size, keeping old_len points
		std::vector<npy_intp> dims(2);
		dims[0] = 2*(new_len-1);
		dims[1] = 3;

		array n_arr = makeNum( dims, (NPY_TYPES)type_npy_traits<CTYPE>::npy_type );
		std::memcpy( cvisual::python::data(n_arr), data(0), sizeof(CTYPE) * old_len * dims[1] );
		array::operator=( n_arr ); // doesn't actually copy

		allocated = dims[0];
	}

	if (new_len > old_len) {
		// Broadcast the last meaningful point over the new points
		(*this)[ slice( old_len, new_len ) ] = (*this)[ slice( old_len-1, old_len ) ];
	}

	length = new_len;
}

template class arrayprim_array<double>;
template class arrayprim_array<float>;

////////////////////////////////

arrayprim::arrayprim()
: count(0)
{
	double* pos_i = pos.data(0);
	for(int i=0; i<3; i++) pos_i[i] = 0;
}

void arrayprim::set_length( size_t new_len ) {
	pos.set_length(new_len);
	count = new_len;
}

object arrayprim::get_pos() {
	return pos[all()];
}

void arrayprim::set_pos( const double_array& n_pos )
{
	std::vector<npy_intp> dims = shape( n_pos );
	if (dims.size() == 1 && !dims[0]) {
		// e.g. pos = ()
		set_length(0);
		return;
	}
	if (dims.size() != 2) {
		throw std::invalid_argument( "pos must be an Nx3 array");
	}
	if (dims[1] == 2) {
		set_length( dims[0] );
		pos[make_tuple(all(), slice(0,2))] = n_pos;
		pos[make_tuple(all(), 2)] = 0.0;
		return;
	}
	else if (dims[1] == 3) {
		set_length( dims[0] );
		pos[all()] = n_pos;
		return;
	}
	else {
		throw std::invalid_argument( "pos must be an Nx3 array");
	}
}

void arrayprim::set_pos_v( const vector& npos ) {
	set_length(1);
	pos[all()] = npos;
}

void arrayprim::set_x( const double_array& arg )
{
	if (shape(arg).size() != 1) throw std::invalid_argument("x must be a 1D array.");
	set_length( shape(arg)[0] );
	pos[make_tuple( all(), 0)] = arg;
}

void arrayprim::set_y( const double_array& arg )
{
	if (shape(arg).size() != 1) throw std::invalid_argument("y must be a 1D array.");
	set_length( shape(arg)[0] );
	pos[make_tuple( all(), 1)] = arg;
}

void arrayprim::set_z( const double_array& arg )
{
	if (shape(arg).size() != 1) throw std::invalid_argument("z must be a 1D array.");
	set_length( shape(arg)[0] );
	pos[make_tuple( all(), 2)] = arg;
}

void arrayprim::set_x_d( const double x)
{
	if (!count)	set_length(1);
	pos[make_tuple( all(), 0)] = x;
}

void arrayprim::set_y_d( const double y)
{
	if (!count)	set_length(1);
	pos[make_tuple( all(), 1)] = y;
}

void arrayprim::set_z_d( const double z)
{
	if (!count)	set_length(1);
	pos[make_tuple( all(), 2)] = z;
}

void arrayprim::append( const vector& npos, int retain )
{
	if (retain > 0 && count >= (size_t)(retain-1))
		set_length(retain-1);		// shifts arrays
	else if (retain == 0)
		set_length(0);
	set_length( count+1);
	double* last_pos = pos.data( count-1 );
	last_pos[0] = npos.x;
	last_pos[1] = npos.y;
	last_pos[2] = npos.z;
}

////////////////////////////////

arrayprim_color::arrayprim_color() {
	double* color_i = color.data(0);
	for(int i=0; i<3; i++) color_i[i] = 1.f;
}

void arrayprim_color::set_length( size_t new_len ) {
	color.set_length(new_len);
	arrayprim::set_length( new_len );
}

object arrayprim_color::get_color() {
	return color[all()];
}

void arrayprim_color::set_color( const double_array& n_color)
{
    std::vector<npy_intp> dims = shape(n_color);
	if (dims.size() == 1 && dims[0] == 3) {
		// A single color, broadcast across the entire (used) array.
		int npoints = (count) ? count : 1;
		color[slice( 0, npoints)] = n_color;
		return;
	}
	if (dims.size() == 2 && dims[1] == 3) {
		// An RGB chunk of color
		set_length(dims[0]);
		color[all()] = n_color;
		return;
	}
	throw std::invalid_argument( "color must be an Nx3 array");
}

void arrayprim_color::set_red( const double_array& arg )
{
	if (shape(arg).size() != 1) throw std::invalid_argument("red must be a 1D array.");
	set_length( shape(arg)[0] );
	color[make_tuple( all(), 0)] = arg;
}

void arrayprim_color::set_green( const double_array& arg )
{
	if (shape(arg).size() != 1) throw std::invalid_argument("green must be a 1D array.");
	set_length( shape(arg)[0] );
	color[make_tuple( all(), 1)] = arg;
}

void arrayprim_color::set_blue( const double_array& arg )
{
	if (shape(arg).size() != 1) throw std::invalid_argument("blue must be a 1D array.");
	set_length( shape(arg)[0] );
	color[make_tuple( all(), 2)] = arg;
}

void arrayprim_color::set_red_d( const double arg )
{
	int npoints = count ? count : 1;
	color[make_tuple(slice(0,npoints), 0)] = arg;
}

void arrayprim_color::set_green_d( const double arg )
{
	int npoints = count ? count : 1;
	color[make_tuple(slice(0,npoints), 1)] = arg;
}

void arrayprim_color::set_blue_d( const double arg )
{
	int npoints = count ? count : 1;
	color[make_tuple(slice(0,npoints), 2)] = arg;
}

void arrayprim_color::append( const vector& npos, const rgb& ncolor, int retain )
{
	append( npos, retain );
	double* last_color = color.data( count-1 );
	last_color[0] = ncolor.red;
	last_color[1] = ncolor.green;
	last_color[2] = ncolor.blue;
}

void arrayprim_color::append_rgb( const vector& npos, double red, double green, double blue, int retain)
{
	append( npos, retain );
	double* last_color = color.data( count-1 );
	if (red != -1) last_color[0] = red;
	if (green != -1) last_color[1] = green;
	if (blue != -1)	last_color[2] = blue;
}

} } // namespace cvisual::python
