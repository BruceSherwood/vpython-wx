// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include <boost/python/detail/wrap_python.hpp>
#include <boost/crc.hpp>

#include "util/errors.hpp"
#include "util/gl_enable.hpp"

#include "python/slice.hpp"
#include "python/extrusion.hpp"

#include <stdexcept>
#include <cassert>
#include <sstream>
#include <iostream>

// Recall that the default constructor for object() is a reference to None.

namespace cvisual { namespace python {

using boost::python::object;
using boost::python::make_tuple;
using boost::python::tuple;

extrusion::extrusion()
	: antialias( true), up(vector(0,1,0)), smooth(0.95),
	  show_start_face(true), show_end_face(true), twosided(true),
	  start(0), end(-1), initial_twist(0.0), center(vector(0,0,0)),
	  first_normal(vector(0,0,0)), last_normal(vector(0,0,0))
{
	scale.set_length(1);
	double* k = scale.data();
	k[0] = 1.0; //scalex
	k[1] = 1.0; //scaley
	k[2] = 0.0; // twist

	contours.insert(contours.begin(), 0.0);
	strips.insert(strips.begin(), 0.0);
	pcontours.insert(pcontours.begin(), 0);
	pstrips.insert(pstrips.begin(), 0);
	normals2D.insert(normals2D.begin(), 0.0);
}

namespace numpy = boost::python::numeric;

//    Serious issue with 32bit vs 64bit machines, apparently,
//    with respect to extract/converting from an array (e.g. double <- int),
//    so for the time being, make sure that in primitives.py one builds
//    the contour arrays as double and int.

void check_array( const array& n_array )
{
	std::vector<npy_intp> dims = shape( n_array );
	if (!(dims.size() == 2 && dims[1] == 2)) {
		throw std::invalid_argument( "This must be an Nx2 array");
	}
}

template <typename T>
void build_contour(const numpy::array& _cont, std::vector<T>& cont)
{
	check_array(_cont);
	std::vector<npy_intp> dims = shape(_cont); 
	size_t length = 2*dims[0];
	cont.resize(length);
	T* start = (T*)data(_cont);
	for(size_t i=0; i<length; i++, start++) {
		cont[i] = *start;
	}
}

void
extrusion::set_contours( const numpy::array& _contours,  const numpy::array& _pcontours,
		const numpy::array& _strips,  const numpy::array& _pstrips  )
{
	// Polygon does not guarantee the winding order of the list of points,
	// so in primitives.py we force the winding order to be clockwise if
	// external and counterclockwise if internal (a hole).

	// primitives.py sends to set_contours descriptions of the 2D surface; see extrusions.hpp
	// We store the information in std::vector containers in flattened form.
	build_contour<npy_float64>(_contours, contours);
	build_contour<npy_int32>(_pcontours, pcontours);
	shape_closed = (bool)pcontours[1];

	if (shape_closed) {
		build_contour<npy_float64>(_strips, strips);
		build_contour<npy_int32>(_pstrips, pstrips);
	}

	size_t ncontours = pcontours[0];
	if (ncontours == 0) return;
	size_t npoints = contours.size()/2; // total number of 2D points in all contours

	maxcontour = 0; // maximum number of points in any of the contours
	for (size_t c=0; c < ncontours; c++) {
		size_t nd = 2*pcontours[2*c+2]; // number of doubles in this contour
		size_t base = 2*pcontours[2*c+3]; // location of first (x) member of 2D (x,y) point
		if (nd/2 > maxcontour) maxcontour = nd/2;
	}

	double xmin, xmax, ymin, ymax; // find outer edges of shape
	xmin = xmax = ymin = ymax = 0.0;
	for (size_t c=0; c < ncontours; c++) {
		size_t nd = 2*pcontours[2*c+2]; // number of doubles in this contour
		size_t base = 2*pcontours[2*c+3]; // location of first (x) member of 2D (x,y) point
		for (size_t pt=0; pt < nd; pt+=2) {
			//double dist = vector(contours[base+pt],contours[base+pt+1],0.).mag();
			double x = contours[base+pt];
			double y = contours[base+pt+1];
			if (x > xmax) xmax = x;
			if (y > ymax) ymax = y;
			if (x < xmin) xmin = x;
			if (y < ymin) ymin = y;
		}
	}

	shape_xmax = std::fabs(xmax);
	xmin = std::fabs(xmin);
	shape_ymax = std::fabs(ymax);
	ymin = std::fabs(ymin);
	if (xmin > shape_xmax) shape_xmax = xmin;
	if (ymin > shape_ymax) shape_ymax= ymin;

	// Set up 2D normals used to build OpenGL triangles.
	// There are two per vertex, because each face of a side of an extrusion segment is a quadrilateral,
	// and each vertex appears in two adjacent triangles.
	// The indexing of normals2D follows that of looping through contour, through point. To use normals2D, you need
	// to use the same nested for-loop structure used here.
	normals2D.resize(4*npoints); // each normal is (x,y), two doubles; there are two normals per vertex (2 tris per vertex)
	size_t i=0;
	for (size_t c=0; c < ncontours; c++) {
		size_t nd = 2*pcontours[2*c+2]; // number of doubles in this contour
		size_t base = 2*pcontours[2*c+3]; // location of first (x) member of 2D (x,y) point
		vector N, Nbefore, Nafter, Navg1, Navg2;
		for (size_t pt=0; pt < nd; pt+=2) {
			if (pt == 0) {
				Nbefore = vector(contours[base+nd-1]-contours[base+1], contours[base]-contours[base+nd-2], 0).norm();
				N =       vector(contours[base+1]-contours[base+3], contours[base+2]-contours[base], 0).norm();
			}
			int after  = (pt+2); // use modulo arithmetic to make the linear sequence effectively circular
			Nafter  = vector(contours[base+((after+1)%nd)]-contours[base+((after+3)%nd)],
					         contours[base+((after+2)%nd)]-contours[base+(after%nd)], 0).norm();
			Navg1 = smoothing(N,Nbefore);
			Navg2 = smoothing(N,Nafter);
			Nbefore = N;
			N = Nafter;
			normals2D[i  ] = Navg1[0];
			normals2D[i+1] = Navg1[1];
			normals2D[i+2] = Navg2[0];
			normals2D[i+3] = Navg2[1];
			i += 4; 
		}
	}
}

vector
extrusion::smoothing(const vector& a, const vector& b) { // vectors a and b need not be normalized
	vector A = a.norm();
	vector B = b.norm();
	if (A.dot(B) > smooth) {
		return (A+B).norm();
	} else {
		return A;
	}
}

void
extrusion::set_antialias( bool aa)
{
	antialias = aa;
}

bool
extrusion::degenerate() const
{
	return count < 2;
}

void
extrusion::set_up( const vector& n_up) {
	up = n_up;
}

shared_vector&
extrusion::get_up()
{
	return up;
}

vector
extrusion::calculate_normal(const vector prev, const vector current, const vector next) {
	// Use 3-point fit to circle to determine final normal (at point "next")
	vector A = (next-current).norm();
	vector lastA = (current-prev).norm();
	double alpha;
	double costheta = lastA.dot(A);
	if (costheta > 1.0) costheta = 1.0; // under certain conditions, costheta was 1+2e-16, alas
	if (costheta < -1.0) costheta = -1.0; // just in case...
	double sintheta = sqrt(1-costheta*costheta);
	if (costheta > smooth && sintheta > 0.0001) { // not a very large or very small bend
		alpha = atan(sintheta/((next-current).mag()/(current-prev).mag() + costheta));
		vector nhat = lastA.cross(A);
		return A.rotate(alpha, nhat).norm();
	} else { // rather abrupt change of direction
		return A;
	}
}

vector
extrusion::get_first_normal()
{
	if (!first_normal) {
		;
	} else {
		return first_normal; // set by user
	}
	vector v0 = vector(0,0,-1);
	if (count == 0) return v0;

	const double* p_i = pos.data();
	vector prev = vector(&p_i[0]);
	vector current, next;
	size_t cnt;
	for (cnt=1; cnt<count; cnt++) {
		current = vector(&p_i[3*cnt]);
		if (!(current-prev)) continue;
		break;
	}
	if (!current) return v0;

	for (cnt++; cnt<count; cnt++) {
		next = vector(&p_i[3*cnt]);
		if (!(next-current)) continue;
		break;
	}
	if (!next) return (prev-current).norm();

	return calculate_normal(next, current, prev);
}

void
extrusion::set_first_normal(const vector& n_first_normal)
{
	// Did not implement this for lack of time. Note however that in the
	// get routine there is a check for whether the normal is (0,0,0), in
	// which case the user has not set the normal. If the user has set the
	// normal, it has to be used in the renderer, which involves not merely
	// making the normal to the face be as specified, but also involves making
	// the angled joint.
	throw std::invalid_argument( "Cannot set first_normal; it is read-only.");
}

vector
extrusion::get_last_normal()
{
	if (!last_normal) {
		;
	} else {
		return last_normal; // set by user
	}
	vector v0 = vector(0,0,1);
	if (count == 0) return v0;

	const double* p_i = pos.data()+3*(count-1);
	vector next = vector(&p_i[0]);
	vector prev, current;
	size_t cnt;
	for (cnt=1; cnt<count; cnt++) {
		current = vector(&p_i[-3*cnt]);
		if (!(next-current)) continue;
		break;
	}
	if (!current) return v0;

	for (cnt++; cnt<count; cnt++) {
		prev = vector(&p_i[-3*cnt]);
		if (!(current-prev)) continue;
		break;
	}
	if (!prev) return (next-current).norm();

	return calculate_normal(prev, current, next);
}

void
extrusion::set_last_normal(const vector& n_last_normal)
{
	// Did not implement this for lack of time. Note however that in the
	// get routine there is a check for whether the normal is (0,0,0), in
	// which case the user has not set the normal. If the user has set the
	// normal, it has to be used in the renderer, which involves not merely
	// making the normal to the face be as specified, but also involves making
	// the angled joint.
	throw std::invalid_argument( "Cannot set last_normal; it is read-only.");
}

void
extrusion::set_length(size_t new_len) {
	scale.set_length(new_len); // this includes twist, the 3rd component of the scale array
	arrayprim_color::set_length(new_len);
}

void
extrusion::appendpos_retain(const vector& n_pos, int retain) {
	if (retain >= 0 && retain < 2)
		throw std::invalid_argument( "Must retain at least 2 points in an extrusion.");
	if (retain > 0 && count >= (size_t)(retain-1))
		set_length(retain-1);		// shifts arrays
	set_length( count+1);
	double* last_pos = pos.data( count-1 );
	last_pos[0] = n_pos.x;
	last_pos[1] = n_pos.y;
	last_pos[2] = n_pos.z;
}

void
extrusion::appendpos_color_retain(const vector& n_pos, const double_array& n_color, const int retain) {
	appendpos_retain(n_pos, retain);
    std::vector<npy_intp> dims = shape(n_color);
	if (dims.size() == 1 && dims[0] == 3) {
		// A single color to be appended.
		color[count-1] = n_color;
		return;
	}
	throw std::invalid_argument( "Appended color must have the form (red,green,blue)");
}

void
extrusion::appendpos_rgb_retain(const vector& n_pos, const double red, const double green, const double blue, const int retain) {
	appendpos_retain(n_pos, retain);
	if (red >= 0) color[count-1][0] = red;
	if (green >= 0) color[count-1][1] = green;
	if (blue >= 0) color[count-1][2] = blue;
}

void
extrusion::set_scale( const double_array& n_scale)
{
	std::vector<npy_intp> dims = shape( n_scale );
	if (dims.size() == 1 && !dims[0]) { // scale=() or [];  reset to size 1
		scale[make_tuple(all(), slice(0,2))] = 1.0;
		return;
	}
	if (dims.size() == 1 && dims[0] == 1) { // scale=[2]
		set_length( dims[0] );
		scale[make_tuple(all(), 0)] = n_scale;
		scale[make_tuple(all(), 1)] = n_scale;
		return;
	}
	if (dims.size() == 1 && dims[0] == 2) { // scale=(2,3) or [2,3]
		set_length( dims[0] );
		scale[make_tuple(all(), slice(0,2))] = n_scale;
		return;
	}
	if (dims.size() == 2 && dims[1] == 2) { // scale=[(2,3),(4,5)....]
		set_length( dims[0] );
		scale[make_tuple(all(), slice(0,2))] = n_scale;
		return;
	}
	else {
		throw std::invalid_argument( "scale must be an Nx2 array");
	}
}

void
extrusion::set_scale_d( const double n_scale)
{
	int npoints = count ? count : 1;
	scale[make_tuple(slice(0,npoints), 0)] = n_scale;
	scale[make_tuple(slice(0,npoints), 1)] = n_scale;
}

boost::python::object extrusion::get_scale() {
	return scale[make_tuple(all(), slice(0,2))];
}

void
extrusion::set_xscale( const double_array& arg )
{
	if (shape(arg).size() != 1) throw std::invalid_argument("xscale must be a 1D array.");
	set_length( shape(arg)[0] );
	scale[make_tuple( all(), 0)] = arg;
}

void
extrusion::set_yscale( const double_array& arg )
{
	if (shape(arg).size() != 1) throw std::invalid_argument("yscale must be a 1D array.");
	set_length( shape(arg)[0] );
	scale[make_tuple( all(), 1)] = arg;
}

void
extrusion::set_xscale_d( const double arg )
{
	int npoints = count ? count : 1;
	scale[make_tuple(slice(0,npoints), 0)] = arg;
}

void extrusion::set_yscale_d( const double arg )
{
	int npoints = count ? count : 1;
	scale[make_tuple(slice(0,npoints), 1)] = arg;
}

void
extrusion::set_twist( const double_array& n_twist)
{
	std::vector<npy_intp> dims = shape( n_twist );
	if (dims.size() == 1 && !dims[0]) { // twist()
		scale[make_tuple(all(), 2)] = 0.0;
		return;
	}
	if (dims.size() == 1 && dims[0] == 1) { // twist(t)
		scale[make_tuple(all(), 2)] = n_twist;
		return;
	}
	if (dims.size() == 1) { // twist(1,2,3)
		set_length( dims[0] );
		scale[make_tuple(all(), 2)] = n_twist;
		return;
	}
	if (dims.size() != 2) {
		throw std::invalid_argument( "twist must be an Nx1 array");
	}
	if (dims[1] == 1) {
		set_length( dims[0] );
		scale[make_tuple(all(), 2)] = n_twist;
		return;
	}
	else {
		throw std::invalid_argument( "twist must be an Nx1 array");
	}
}

void
extrusion::set_twist_d( const double n_twist)
{
	int npoints = count ? count : 1;
	scale[make_tuple(slice(0,npoints), 2)] = n_twist;
}

boost::python::object extrusion::get_twist() {
	return scale[make_tuple(all(), 2)];
}
void
extrusion::set_initial_twist(const double n_initial_twist) {
	initial_twist = n_initial_twist;
}

double
extrusion::get_initial_twist(){
	return initial_twist;
}

void
extrusion::set_start(const int n_start) {
	start = n_start;
}

int
extrusion::get_start(){
	return start;
}

void
extrusion::set_end(const int n_end){
	end = n_end;
}

int
extrusion::get_end() {
	return end;
}

void
extrusion::set_twosided(const bool n_twosided){
	twosided = n_twosided;
}

bool
extrusion::get_twosided() {
	return twosided;
}

void
extrusion::set_show_start_face(const bool n_show_start_face) {
	show_start_face = n_show_start_face;
}

bool
extrusion::get_show_start_face(){
	return show_start_face;
}

void
extrusion::set_show_end_face(const bool n_show_end_face){
	show_end_face = n_show_end_face;
}

bool
extrusion::get_show_end_face() {
	return show_end_face;
}

void
extrusion::set_smooth(const double n_smooth){
	smooth = n_smooth;
}

double
extrusion::get_smooth() {
	return smooth;
}

bool
extrusion::monochrome(double* tcolor, size_t pcount)
{
	rgb first_color( tcolor[0], tcolor[1], tcolor[2]);
	size_t nn;

	for(nn=0; nn<pcount; nn++)  {
		if (tcolor[nn*3] != first_color.red)
			return false;
		if (tcolor[nn*3+1] != first_color.green)
			return false;
		if (tcolor[nn*3+2] != first_color.blue)
			return false;
	}
	return true;
}

void
extrusion::gl_pick_render(view& scene)
{
	// TODO: Should be able to pick an extrusion, presumably. Old comment about curves:
	// Aack, I can't think of any obvious optimizations here.
	// But since Visual 3 didn't permit picking of curves, omit for now.
	// We can't afford it; serious impact on performance.
	//gl_render( scene);
}

vector
extrusion::get_center() const
{
	// Apparently this never gets called.
	// Below is code that was started on the assumption that this does get called.
	// After writing that code, different code was added into the render code.
	return center;
	/*
	double xmin, xmax, ymin, ymax; // outer edges of shape
	xmin = xmax = ymin = ymax = 0.0;
	size_t ncontours = pcontours[0];
	for (size_t c=0; c < ncontours; c++) {
		size_t nd = 2*pcontours[2*c+2]; // number of doubles in this contour
		size_t base = 2*pcontours[2*c+3]; // location of first (x) member of 2D (x,y) point
		for (size_t pt=0; pt < nd; pt+=2) {
			//double dist = vector(contours[base+pt],contours[base+pt+1],0.).mag();
			double x = contours[base+pt];
			double y = contours[base+pt+1];
			if (x > xmax) xmax = x;
			if (y > ymax) ymax = y;
			if (x < xmin) xmin = x;
			if (y < ymin) ymin = y;
		}
	}
	vector ret = vector(0,0,0);
	if (count == 0) return vector((xmin+xmax)/2., (ymin+ymax)/2., 0.0);
	const double* pos_i = pos.data();
	const double* pos_end = pos.end();
	const double* s_i = scale.data();
	double maxscale = 0.0;
	vector lastA = vector(0,0,0);
	vector A = lastA;
	for (size_t i=0; pos_i < pos_end; i++, pos_i += 3, s_i += 3) {
		double scalex = s_i[0];
		double scaley = s_i[1];
		vector current = vector(&pos_i[0]);
		if (i == count-1) {
			A = lastA;
		} else {
			A = vector(&pos_i[3])-current;
		}
		ret.x += pos_i[0]+scalex*xmax;
		ret.x += pos_i[0]+scaley*xmin;
		ret.y += pos_i[1]+scaley*ymax;
		ret.y += pos_i[1]+scaley*ymin;
		ret.z += pos_i[2];
		ret.z += pos_i[2];
	}
	return ret/(2*count);
	*/
}

void
extrusion::grow_extent( extent& world)
{
	maxextent = 0.0; // maximum scaled distance from curve
	size_t istart, iend;
	const double* pos_i = pos.data();
	const double* s_i = scale.data();
	if (count == 0) {
		istart = 0;
	} else {
		if (start < 0) {
			if (((int)count+start) < 0) {
				return; // nothing to display
			} else {
				istart = int(count)+start;
			}
		} else {
			istart = start;
		}
		if (istart > count-1) return; // nothing to display
	}

	if (count == 0) {
		iend = 0;
	} else {
		if (end < 0) {
			if (((int)count+end) < 0) {
				return; // nothing to display
			} else {
				iend = int(count)+end;
			}
		} else {
			iend = end;
		}
		if (iend < startcorner) return; // nothing to display
	}
	pos_i += 3*istart;
	s_i += 3*istart;
	if (count == 0) { // just show shape
		world.add_sphere(vector(0,0,0), std::max(shape_xmax*scale.data()[0],shape_ymax*scale.data()[1]));
	} else {
		for (size_t i=istart; i <= iend; i++, pos_i+=3, s_i+=3) {
			double xmax = s_i[0]*shape_xmax;
			double ymax = s_i[1]*shape_ymax;
			if (ymax > xmax) xmax = ymax;
			if (xmax > maxextent) maxextent = xmax;
			world.add_sphere( vector(pos_i), xmax);
		}
	}
	world.add_body();
}

bool
extrusion::adjust_colors( const view& scene, double* tcolor, size_t pcount)
{
	rgb rendered_color;
	bool mono = monochrome(tcolor, pcount);
	if (mono) {
		// We can get away without using a color array.
		rendered_color = rgb( tcolor[0], tcolor[1], tcolor[2]);
		if (scene.anaglyph) {
			if (scene.coloranaglyph)
				rendered_color.desaturate().gl_set(opacity);
			else
				rendered_color.grayscale().gl_set(opacity);
		}
		else
			rendered_color.gl_set(opacity);
	}
	else {
		glEnableClientState( GL_COLOR_ARRAY);
		if (scene.anaglyph) {
			// Must desaturate or grayscale the color.

			for (size_t i = 0; i < pcount; ++i) {
				rendered_color = rgb( tcolor[3*i], tcolor[3*i+1], tcolor[3*i+2]);
				if (scene.coloranaglyph)
					rendered_color = rendered_color.desaturate();
				else
					rendered_color = rendered_color.grayscale();
				tcolor[3*i] = rendered_color.red;
				tcolor[3*i+1] = rendered_color.green;
				tcolor[3*i+2] = rendered_color.blue;
			}
		}
	}
	return mono;
}

// There were unsolvable problems with rotate. See comments with the extrude routine.
/*
void
extrusion::rotate( double angle, const vector& _axis, const vector& origin)
{

	tmatrix R = rotation( angle, _axis, origin);
	double* p_i = pos.data();
	for (size_t i = 0; i < count; i++, p_i += 3) {
		vector temp = R*vector(&p_i[0]);
		p_i[0] = temp[0];
		p_i[1] = temp[1];
		p_i[2] = temp[2];
	}
	if (!_axis.cross(up)) return;
	up = R.times_v(up);
}
*/

void
extrusion::gl_render(view& scene)
{
	std::vector<vector> faces_pos;
	std::vector<vector> faces_normals;
	std::vector<vector> faces_colors;
	clear_gl_error();
	gl_enable_client vertex_arrays( GL_VERTEX_ARRAY);
	gl_enable_client normal_arrays( GL_NORMAL_ARRAY);
	gl_enable_client colors( GL_COLOR_ARRAY);
	gl_enable cull_face( GL_CULL_FACE);
	extrude(scene, faces_pos, faces_normals, faces_colors, false);
	glDisableClientState( GL_VERTEX_ARRAY);
	glDisableClientState( GL_NORMAL_ARRAY);
	glDisableClientState( GL_COLOR_ARRAY);
	check_gl_error();
}

boost::python::object extrusion::_faces_render() {
	// Mock up scene machinery:
	gl_extensions glext;
	double gcf = 1.0;
	view scene( vector(0,0,1), vector(0,0,0), 400,
		400, false, gcf, vector(gcf,gcf,gcf), false, glext);
	std::vector<vector> faces_pos;
	std::vector<vector> faces_normals;
	std::vector<vector> faces_colors;
	extrude( scene, faces_pos, faces_normals, faces_colors, true);
	std::vector<npy_intp> dimens(2);
	size_t d = faces_pos.size(); // number of pos vectors (3*d doubles)
	dimens[0] = 3*d; // make array of vectors 3d long (pos, normals, colors)
	dimens[1] = 3;
	array faces_data = makeNum(dimens);
	memmove( data(faces_data), &faces_pos[0], sizeof(vector)*d );
	memmove( data(faces_data)+sizeof(vector)*d, &faces_normals[0], sizeof(vector)*d );
	memmove( data(faces_data)+sizeof(vector)*2*d, &faces_colors[0], sizeof(vector)*d );
	return faces_data;
}

void
extrusion::render_end(const vector V, const vector current,
		const double c11, const double c12, const double c21, const double c22,
		const vector xrot, const vector y, const vector current_color, bool show_first,
		std::vector<vector>& faces_pos,
		std::vector<vector>& faces_normals,
		std::vector<vector>& faces_colors, bool make_faces)
{
	// if (make_faces && show_first), make the first set of triangles else make the second set

	// Use the triangle strips in "strips" to paint an end of the extrusion
	size_t npstrips = pstrips[0]; // number of triangle strips in the cross section
	size_t spoints = strips.size()/2; // total number of 2D points in all strips
	double tx, ty;

	for (size_t c=0; c<npstrips; c++) {
		size_t nd = 2*pstrips[2*c+2]; // number of doubles in this strip
		size_t base = 2*pstrips[2*c+3]; // initial (x,y) = (strips[base], strips[base+1])
		std::vector<vector> tristrip(nd/2), snormals(nd/2), endcolors(nd/2);

		for (size_t pt=0, n=0; pt<nd; pt+=2, n++) {
			tx = c11*strips[base+pt] + c12*strips[base+pt+1];
			ty = c21*strips[base+pt] + c22*strips[base+pt+1];
			tristrip[n] = current + tx*xrot + ty*y;
			snormals[n] = V;
			endcolors[n] = current_color;
		}

		if (!make_faces && (show_first || twosided)) {
			glNormalPointer( GL_DOUBLE, 0, &snormals[0]);
			glVertexPointer(3, GL_DOUBLE, 0, &tristrip[0]);
			glColorPointer(3, GL_DOUBLE, 0, &endcolors[0]);
			// nd doubles, nd/2 vertices
			glDrawArrays(GL_TRIANGLE_STRIP, 0, nd/2);
		} else if (make_faces && show_first){
			for (size_t pt=0, n=0; pt<(nd-4); pt+=2, n++) {
				faces_normals.insert(faces_normals.end(), snormals.begin()+n, snormals.begin()+n+3);
				faces_colors.insert(faces_colors.end(), endcolors.begin()+n, endcolors.begin()+n+3);
				if (n % 2) { // if odd
					faces_pos.push_back(tristrip[n]);
					faces_pos.push_back(tristrip[n+2]);
					faces_pos.push_back(tristrip[n+1]);
				} else {
					faces_pos.insert(faces_pos.end(), tristrip.begin()+n, tristrip.begin()+n+3);
				}
			}
		}

		// Make two-sided:
		for (size_t pt=0, n=0; pt<nd; pt+=2, n++) {
			size_t nswap;
			if (n % 2) { // if odd
				nswap = n-1;
			} else {
				if (pt == nd-2) { // total number of points is odd
					nswap = n;
				} else {
					nswap = n+1;
				}
			}
			tx = c11*strips[base+pt] + c12*strips[base+pt+1];
			ty = c21*strips[base+pt] + c22*strips[base+pt+1];
			tristrip[nswap] = current + tx*xrot + ty*y;
			snormals[n] = -V;
		}

		if (!make_faces && (!show_first || twosided)) {
			// nd doubles, nd/2 vertices
			glDrawArrays(GL_TRIANGLE_STRIP, 0, nd/2);
		} else if (make_faces && !show_first){
			for (size_t pt=0, n=0; pt<(nd-4); pt+=2, n++) {
				faces_normals.insert(faces_normals.end(), snormals.begin()+n, snormals.begin()+n+3);
				faces_colors.insert(faces_colors.end(), endcolors.begin()+n, endcolors.begin()+n+3);
				if (n % 2) { // if odd
					faces_pos.push_back(tristrip[n]);
					faces_pos.push_back(tristrip[n+2]);
					faces_pos.push_back(tristrip[n+1]);
				} else {
					faces_pos.insert(faces_pos.end(), tristrip.begin()+n, tristrip.begin()+n+3);
				}
			}
		}
	}
}

	/* The following code converts the triangle strips to triangles. For a big text object it was half as fast.
	// The code is being saved here for the time being on the chance it would be useful in the case of wrapping an extrusion.
	for (size_t c=0; c<npstrips; c++) {
		size_t nd = 2*pstrips[2*c+2]; // number of doubles in this strip
		size_t base = 2*pstrips[2*c+3]; // initial (x,y) = (strips[base], strips[base+1])
		std::vector<vector> v(nd/2);

		for (size_t pt=0, n=0; pt<nd; pt+=2, n++) {
			tx = c11*strips[base+pt] + c12*strips[base+pt+1];
			ty = c21*strips[base+pt] + c22*strips[base+pt+1];
			v[n] = current + tx*xrot + ty*y; // create array of nd/2 3D vector strip locations
		}

		if (!make_faces) {
			faces_pos.reserve(nd/2-2);
			faces_normals.reserve(nd/2-2);
			faces_colors.reserve(nd/2-2);
			faces_pos.clear();
			faces_normals.clear();
			faces_colors.clear();
		}

		if (show_first || (!make_faces && twosided)){
			for (size_t n=0; n<(nd/2-2); n++) {
				faces_normals.push_back(V);
				faces_normals.push_back(V);
				faces_normals.push_back(V);
				faces_colors.push_back(current_color);
				faces_colors.push_back(current_color);
				faces_colors.push_back(current_color);
				if (n % 2) { // if odd
					faces_pos.push_back(v[n  ]);
					faces_pos.push_back(v[n+2]);
					faces_pos.push_back(v[n+1]);
				} else {
					faces_pos.insert(faces_pos.end(), v.begin()+n, v.begin()+n+3);
				}
			}
			if (!make_faces) {
				glNormalPointer( GL_DOUBLE, 0, &faces_normals[0]);
				glVertexPointer(3, GL_DOUBLE, 0, &faces_pos[0]);
				glColorPointer(3, GL_DOUBLE, 0, &faces_colors[0]);
				glDrawArrays(GL_TRIANGLES, 0, faces_pos.size());
				faces_pos.clear();
				faces_normals.clear();
				faces_colors.clear();
			}
		}

		if (!show_first || (!make_faces && twosided)){
			for (size_t n=0; n<(nd/2-2); n++) {
				faces_normals.push_back(-V);
				faces_normals.push_back(-V);
				faces_normals.push_back(-V);
				faces_colors.push_back(current_color);
				faces_colors.push_back(current_color);
				faces_colors.push_back(current_color);
				if (n % 2) { // if odd
					faces_pos.insert(faces_pos.end(), v.begin()+n, v.begin()+n+3);
				} else {
					faces_pos.push_back(v[n  ]);
					faces_pos.push_back(v[n+2]);
					faces_pos.push_back(v[n+1]);
				}
			}
			if (!make_faces) {
				glNormalPointer( GL_DOUBLE, 0, &faces_normals[0]);
				glVertexPointer(3, GL_DOUBLE, 0, &faces_pos[0]);
				glColorPointer(3, GL_DOUBLE, 0, &faces_colors[0]);
				glDrawArrays(GL_TRIANGLES, 0, faces_pos.size());
			}
		}
	}
}
*/

void
extrusion::extrude( const view& scene,
		std::vector<vector>& faces_pos,
		std::vector<vector>& faces_normals,
		std::vector<vector>& faces_colors, bool make_faces)
{
	// TODO: A twist of 0.1 shows surface breaks, even with very small smooth....?

	// The basic architecture of the extrusion object:
	// Use the Polygon module to create by constructive geometry a 2D surface in the form of a
	// set of contours, each a sequence of 2D points. In primitives.py these contours are forced to
	// be ordered CW if external and CCW if internal (holes). In set_contours, 2D normals to these
	// contours are calculated, with smoothing around the contour. The 2D surface, including the
	// computed normals, is extruded as a cross section along a curve defined by the pos attribute,
	// which like other array objects (curve, points, faces, convex) is represented by a numpy array.

	// For efficiency, orthogonal unit vectors (xaxis,yaxis) in the 2D surface are defined so that a
	// contour point (a,b) relative to the curve is a vector in the plane a*xaxis+b*yaxis.
	// At a joint between adjacent extrusion segments, from (xaxis,yaxis) that are perpendicular
	// to the curve, we derive new unit vectors (x,y) in the plane of 2D surface such that y is
	// parallel to the "axle" of rotation of the joint, the locus of points for which no rotation
	// occurs. We determine coefficients such that a*xaxis+b*yaxis = aa*x+bb*y, so that
	// aa = c11*a + c12*b and bb = c21*a + c22*b. We then create a non-unit vector xrot in the
	// plane of the joint, perpendicular to y, so that positions in the joint corresponding to (a,b)
	// are given by aa*xrot + bb*y. These joint positions are connected to the previous joint positions
	// to form one segment of the extrusion. (We don't save all the previous positions; rather we simply
	// save the previous values of (x,y) from which the previous positions can easily be calculated.)

	// This was how normals were originally calculated, but we've changed to a different scheme:
		// The normals to the sides of the extrusion are simply computed from the (nx,ny) normals,
		// computed in set_contours, as nx*xaxis+ny*yaxis. By look-ahead, using what (xaxis,yaxis) will
		// be in the next segment, the normals at the end of one segment are smoothed to normals at the
		// start of the next segment. At the start of the next segment we look behind to smooth the normals.

	// Consider somewhere along the path points r1, r2, r3 and define theta as the angle between (r3-r2) and
	// (r2-r1). If this angle is large, with cosine given by (r3-r2.dot(r2-r1) being less than "smooth" (default 0.95),
	// the normal to the joint at r2 is halfway between (r3-r2) and (r2-r1). If however the angle is small, with cosine
	// greater than smooth, fit a circle to the three points, if possible (if the three points
	// are in a straight line, the normal is just the direction of that line). Then the normal to the joint at
	// r2 is rotated an angle alpha from the direction of (r2-r1), in the direction of theta, and alpha
	// is obtained from tan(alpha) = sin(theta)/( |r3-r2|/|r2-r1| + cos(theta) ). If r1 is the first point in
	// the path, the normal to the initial face is rotated -alpha from the direction of (r2-r1).
	// At the end of the path, the normal to the final face is rotated away from (r[-1]-r[-2]) by the negative
	// of the angle alpha by which the normal to point r[-2] was rotated relative to (r[-2]-r[-3]), unless of
	// course the last direction change is greater than what is smoothed, or there is a straight line.
	// Note that a straight line is characterized by sin(theta) == 0.

	// Using the scale array of (scalex,scaley) values, the actual position of a point (a,b) in the
	// 2D surface is given by (scalex*a, scaley*b). Note that the precalculated normal to a vector (a,b)
	// on the 2D surface is in the direction (-b, a), as can be seen from the dot product being zero.
	// This means that when nonuniform scaling is in effect (scalex != scaley), the direction of a normal
	// changes to be in the direction (-scaley*b, scalex*a).

	// The scale array is an array of (scalex, scaley, twist), where twist is the angle (in radians) of
	// the CCW rotation from the previous point. The twist for the initial point is ignored for convenience.
	// An initial twist can be set with initial_twist, which is often more convenient than setting "up".

	// extrusion.shape=Polygon(...) not only sends contour information to set_contours but also
	// sends triangle strips used to render the front and back surfaces of the extrusion.

	// shape can be a non-closed contour, in which case we generate a "skin" (zero thickness, no end faces).
	// Polygon deals with closed contours but doesn't always add the initial point to the last point.
	// primitives.py set_shape distinguishes between a (possibly multicontour) Polygon object and a
	// simple list of points. For a Polygon object, it deletes an unnecessary final point that is equal
	// to the initial point, because the rendering in that case will generate two extra triangles, but this
	// is marked as "closed" (pcontours[1]=1). For a simple list of points, if final == initial we discard the final
	// point and mark this as "closed", but if final != initial we mark this as "not closed".

	// It may have been unavoidable, but there's a fair amount of complexity in the sequencing of the
	// rendering of the various components (initial face, segments, final face), depending on whether
	// the path is closed or not. If closed, it is possible to determine the geometry of the joint at
	// pos[0] because pos[0] lies between pos[-2] and pos[1], with pos[-1] = pos[0], so that pos[0] is
	// the middle point on a possibly circular arc determined by pos[-2], pos[0], and pos[1]. If the
	// path is not closed, final determination of the geometry at pos[0] must await geometry calculations
	// at pos[1]. If pos[0], pos[1], and pos[2] lie on a circle with small bending, the front face at
	// pos[0] is angled to be radial to the circle; otherwise the front face is perpendicular to the
	// direction of pos[1]-pos[0].

	// Bug with no resolution: The following program displays only the red back face.
	// Remove the E.rotate statement and all of the simple box-like extrusion is displayed.
	// Putting std::cout outputs into the code and running in the Windows command window
	// shows that all of the appropriate code is being executed, so why are the other portions
	// of the object missing? If the rotation is around (0,0,1) instead of (1,0,0) or (0,1,0)
	// the entire object displays. Also, rotation around (x,y,1) fails if either x or y or both
	// are nonzero. A variant on this program is to follow this with a loop that continually
	// rotates the object, which flickers between showing the entire object and showing only
	// the back face. If the extrusion is put in a frame and the frame rotated, it also
	// flickers. But in that case, if the single E.rotate statement is removed, rotation of
	// the frame shows no problems. Note that the sequence of rendering for this simple
	// object is front face, back face, 0 to 1 segment. Very strange....
	// r = shapes.rectangle(width=10)
	// c = [color.cyan, color.red]
	// E = extrusion(pos=[(0,0,-1), (0,0,-9)], shape=r, color=c)
	// E.rotate(angle=0.3, axis=(1,0,0), origin=(0,0,-5))
	// As a result, the rotate method has been removed from extrusions. As with the other array
	// objects (curve, points, faces, convex), put the extrusion in a frame and rotate the frame.

	const int LINE_LENGTH = 1000; // The maximum number of points to display.
	// Data storage for the position and color data (plus room for extra point in the case of a closed contour)
	double spos[3*(LINE_LENGTH+3+3+3)]; // room for extra point if startcorner and endcorner in same step
	double tcolor[3*(LINE_LENGTH+3+3+3)];
	float tscale[3*(LINE_LENGTH+3+3+3)]; // scale factors, and twist
	float fstep = (float)(count-1)/(float)(LINE_LENGTH-1);
	if (fstep < 1.0F) fstep = 1.0F;
	size_t iptr=0, iptr3, pcount=0;

	const double* v_i = pos.data();
	const double* cd_i = color.data();
	const double* sd_i = scale.data();

	if (count == 0) {
		startcorner = 0;
	} else {
		if (start < 0) {
			if (((int)count+start) < 0) {
				return; // nothing to display
			} else {
				startcorner = int(count)+start;
			}
		} else {
			startcorner = start;
		}
		if (startcorner > count-1) return; // nothing to display
	}

	if (count == 0) {
		endcorner = 0;
	} else {
		if (end < 0) {
			if (((int)count+end) < 0) {
				return; // nothing to display
			} else {
				endcorner = int(count)+end;
			}
		} else {
			endcorner = end;
		}
		if (endcorner < startcorner) return; // nothing to display
	}

	if (count < 1) {
		pcount = 1;
		spos[0] = spos[1] = spos[2] = 0.0;
		tcolor[0] = cd_i[0];
		tcolor[1] = cd_i[1];
		tcolor[2] = cd_i[2];
		tscale[0] = sd_i[0];
		tscale[1] = sd_i[1];
		tscale[2] = sd_i[2];
	} else {
		size_t tstart = startcorner;
		size_t tend = endcorner;
		size_t stepsize = (int)(fstep+.5);
		// Choose which points to display
		for (float fptr=0.0; iptr < count && pcount < LINE_LENGTH; fptr += fstep, iptr = (int)(fptr+.5), ++pcount) {
			size_t startoffset = 0;
			size_t endoffset = 0;
			// Make sure that startcorner is in the display set
			if (stepsize > 1 && startcorner >= iptr && startcorner < iptr+stepsize) {
				startoffset = startcorner-iptr;
				tstart = pcount;
			}

			iptr3 = 3*(iptr+startoffset);
			spos[3*pcount  ] = scene.gcf*v_i[iptr3];
			spos[3*pcount+1] = scene.gcf*v_i[iptr3+1];
			spos[3*pcount+2] = scene.gcf*v_i[iptr3+2];
			tcolor[3*pcount  ] = cd_i[iptr3];
			tcolor[3*pcount+1] = cd_i[iptr3+1];
			tcolor[3*pcount+2] = cd_i[iptr3+2];
			tscale[3*pcount  ] = sd_i[iptr3];
			tscale[3*pcount+1] = sd_i[iptr3+1];
			tscale[3*pcount+2] = sd_i[iptr3+2];

			// Make sure that endcorner is in the display set
			if (stepsize > 1 && endcorner >= iptr && endcorner < iptr+stepsize) {
				endoffset = endcorner-iptr;
				if (endcorner == startcorner) {
					// Already have this point
					tend = tstart;
				} else if (iptr == count-1) {
					tend = pcount;
				} else {
					iptr3 = 3*(iptr+endoffset);
					spos[3*pcount  ] = scene.gcf*v_i[iptr3];
					spos[3*pcount+1] = scene.gcf*v_i[iptr3+1];
					spos[3*pcount+2] = scene.gcf*v_i[iptr3+2];
					tcolor[3*pcount  ] = cd_i[iptr3];
					tcolor[3*pcount+1] = cd_i[iptr3+1];
					tcolor[3*pcount+2] = cd_i[iptr3+2];
					tscale[3*pcount  ] = sd_i[iptr3];
					tscale[3*pcount+1] = sd_i[iptr3+1];
					tscale[3*pcount+2] = sd_i[iptr3+2];
					pcount += 1;
					tend = pcount;
				}
			}
		}
		startcorner = tstart;
		endcorner = tend;
	}

	size_t ncontours = pcontours[0];
	if (ncontours == 0) return;
	size_t npoints = contours.size()/2; // total number of 2D points in all contours

	// 3 positions and normals per triangle, and the number of triangles = 2 times the number of points in the 2D shape,
	// times 2 for front and back of each triangle. Allocate space to hold the largest contour:
	size_t maxtriangles = 12*maxcontour;
	std::vector<vector> tris(maxtriangles), normals(maxtriangles), tcolors(maxtriangles);

	vector xaxis, yaxis; // local unit-vector axes on the 2D shape
	vector prevxaxis, prevyaxis; // local unit-vector axes on the 2D shape on preceding segment
	vector nextxaxis, nextyaxis; // local unit-vector axes on the 2D shape on following segment
	vector prevx, prevy; // local axes on previous plane perpendicular to curve
	vector prevxrot; // local axis in the plane of the joint
	bool smoothed; // true if bend not large
	bool prevsmoothed; // true if previous bend not large
	double prevc11, prevc12, prevc21, prevc22; // previous rotation coefficients on the 2D surface
	double alpha = 0.0; // rotation of bisecting_plane_normal
	double theta; // angle of rotation from lastA to A
	double lastalpha; // previous rotation of bisecting_plane_normal
	// vector extcenter = vector(0,0,0); // the geometric center of the extrusion (apparently not used)

	// pos and color iterators
	v_i = spos;
	const double* c_i = tcolor;
	const float* s_i = tscale;

	vector current_color = vector(tcolor[0], tcolor[1], tcolor[2]);
	vector prev_color = current_color;
	const vector initial_face_color = vector(c_i[0], c_i[1], c_i[2]);
	const vector final_face_color = vector(c_i[3*(pcount-1)], c_i[3*(pcount-1)+1], c_i[3*(pcount-1)+2]);

	if (!make_faces) bool mono = adjust_colors( scene, tcolor, pcount);
	bool closed = false;
	if (pcount > 2) {
		double path_length = 0.0;
		double* p=spos;
		for (size_t n=0; n<(pcount-1); n++, p+=3) {
			path_length += (vector(&p[3]) - vector(&p[0])).mag();
		}
		closed = ((vector(&spos[0]) - vector(&spos[(pcount-1)*3])).mag() <= 0.0001*path_length);
	}
	bool show_start = show_start_face;
	bool show_end = show_end_face;
	if (!shape_closed || (closed && startcorner == 0 && endcorner == pcount-1))
		show_start = show_end = false;

	bool zerodepth = false; // true if should display just one 2D surface

	vector A; // points from previous point to current point
	vector lastA = vector(0,0,0); // unit vector of previous segment
	vector prev = vector(&spos[0]); // previous location; this probably isn't quite right
	vector current; // point along the curve currently being processed
	vector next; // next location (so sequence is prev, current, next)

	// Note that pcount is never zero; gl_render mocks up at least one point.

	if (closed) {
		// find previous non-duplicate point
		size_t ending = pcount-1;
		size_t pt;
		for (pt=ending; pt >= 0; pt--) {
			lastA = vector(&spos[3*pt]) - vector(&spos[3*pt-3]);
			if (!lastA) {
				continue;
			}
			lastA = lastA.norm();
			prev = vector(&spos[3*pt-3]);
			break;
		}
		// find next non-duplicate point
		for (pt=0; pt <= ending; pt++) {
			A = vector(&spos[3*pt+3]) - vector(&spos[3*pt]);
			if (!A) {
				continue;
			}
			next = vector(&spos[3*pt+3]);
			A = A.norm();
			break;
		}
		prevsmoothed = (A.dot(lastA) > smooth);
		// add another point to a closed curve, equal to point 1
		spos[3*pcount]   = spos[3*pt+3];
		spos[3*pcount+1] = spos[3*pt+4];
		spos[3*pcount+2] = spos[3*pt+5];
		tcolor[3*pcount]   = tcolor[3*pt+3];
		tcolor[3*pcount+1] = tcolor[3*pt+4];
		tcolor[3*pcount+2] = tcolor[3*pt+5];
		tscale[3*pcount]   = tscale[3*pt+3];
		tscale[3*pcount+1] = tscale[3*pt+4];
		tscale[3*pcount+2] = tscale[3*pt+5];
	}

	size_t lastpoint = pcount-1;

	if (make_faces) {
		// Calculate the total number of triangles
		// contours.size()/2 is number of vertices in the 2D shape
		// On the sides of the extrusion there are 2 single-sided triangles for every contour vertex
		size_t triangles = contours.size()*(endcorner-startcorner); // single-sided triangles on the extrusion sides
		// strips.size()/2 is the number of vertices in triangle strips on an end face, and there are pstrips[0] strips
		// If there are N vertices in a triangle strip, there are N-2 single-sided triangles, so there are
		//    strips.size()/2 - 2*pstrips[0] single-sided triangles on an end face
		size_t endtriangles = strips.size()/2-2*pstrips[0]; // single-sided triangles on the extrusion ends
		if (show_start_face && (!closed || (startcorner > 0))) triangles += endtriangles;
		if (show_end_face && (!closed || (endcorner < lastpoint))) triangles += endtriangles;
		faces_pos.reserve(3*triangles); // 3D vectors
		faces_normals.reserve(3*triangles); // 3D vectors
		faces_colors.reserve(3*triangles); // 3D vectors
	}

	bool delay_initial_face = false; // True if delay rendering of initial face (waiting for normal info)

	bool rendered_initial_face = false; // True when processing the first corner after the beginning

	for (size_t corner = 0; corner <= endcorner; ++corner, v_i += 3, c_i += 3, s_i += 3) {
		size_t icorner = corner;
		current = vector(&v_i[0]);
		current_color = vector(c_i[0], c_i[1], c_i[2]);

		// A is a unit vector pointing from the current location to the next location along the curve.
		// lastA is a unit vector pointing from the previous location to the current location.

		// Skip over duplicate points (but retain correct color information)
		const double* tv_i = v_i;
		for (size_t tcorner=corner; tcorner <= endcorner; ++tcorner, tv_i += 3) {
			if (corner == startcorner) {
				current_color = vector(c_i[0], c_i[1], c_i[2]);
			}
			if (!closed && (corner == lastpoint)) {
				// if (!closed && corner == 0), lastA == (0,0,0) by initialization of lastA.
				A = lastA;
				next = current;
				break;
			} else {
				next = vector( &tv_i[3]);
				A = (next-current).norm();
				if (icorner == 0 && !closed) lastA = A; // lastA already set for closed path
			}
			if (!A) {
				v_i += 3;
				c_i += 3;
				s_i += 3;
				corner++;
				continue;
			} else {
				break;
			}
		}
		if (!A) {
			zerodepth = true;
			if (!lastA) {
				A = lastA = vector(0,0,-1);
			} else {
				A = lastA;
			}
		}

		// Calculate the normal to the plane which is the intersection of adjacent segments:
		vector bisecting_plane_normal = (A + lastA).norm();
		double costheta = lastA.dot(A);
		if (costheta > 1.0) costheta = 1.0; // under certain conditions, costheta was 1+2e-16, alas
		if (costheta < -1.0) costheta = -1.0; // just in case...
		theta = acos(costheta);
		double sintheta = sqrt(1-costheta*costheta);
		if (costheta > smooth && sintheta > 0.0001) { // not a very large or very small bend
			smoothed = true;
			alpha = atan(sintheta/((next-current).mag()/(current-prev).mag() + costheta));
			vector nhat = lastA.cross(A);
			bisecting_plane_normal = lastA.rotate(alpha, nhat).norm();
		} else {
			smoothed = false;
			double dotprod = lastA.dot(bisecting_plane_normal);
			if (dotprod > 1.) {
				alpha = 0;
			} else if (dotprod < -1.0) {
				alpha = acos(-1.0);
			} else {
				alpha = acos(dotprod);
			}
		}

		if (!bisecting_plane_normal) {  //< Exactly 180 degree bend
			bisecting_plane_normal = vector(0,0,1).cross(A);
			if (!bisecting_plane_normal)
				bisecting_plane_normal = vector(0,1,0).cross(A);
		}

		if (icorner == 0) {
			// On the 2D surface position (a,b) is at a*xaxis+b*yaxis, where
			//    xaxis=(1,0) and yaxis=(0,1) on that 2D surface.
			yaxis = up;
			xaxis = A.cross(yaxis).norm();
			if (!xaxis) xaxis = A.cross( vector(0, 0, 1)).norm();
			if (!xaxis) xaxis = A.cross( vector(1, 0, 0)).norm();
			yaxis = xaxis.cross(A).norm();

			if (!xaxis || !yaxis || !(xaxis-yaxis)) {
				std::ostringstream msg;
				msg << "Degenerate extrusion case! please report the following "
					"information to visualpython-users@lists.sourceforge.net: ";
				msg << "current:" << current
			 		<< " A:" << A << " x:" << xaxis << " y:" << yaxis
			 		<< std::endl;
			 	VPYTHON_WARNING( msg.str());
			}
		}

		// A point (a,b) in the 2D surface is located in 3D space at current+a*xaxis+b*yaxis.
		// Re-express points in the 2D surface in terms of (x,y) axes, where y is parallel
		// to the "axle" of rotation of the joint, along which points don't rotate.
		vector x;
		vector y = (lastA.cross(A)).norm(); // parallel to the axle of the joint
		if (!y) {
			y = yaxis; // the joint is perpendicular to the extrusion segment; no change of direction
			x = xaxis;
		} else {
			if (icorner == 0 && closed) {
				x = A.cross(y);
			} else {
				x = lastA.cross(y);
			}
		}

		// If twist is positive, rotate xaxis and yaxis CCW
		if (s_i[2] || (icorner == 0 && initial_twist)) {
			double angle;
			if (icorner) {
				angle = s_i[2];
			} else {
				angle = initial_twist; // ignore twist[0]; use initial_twist instead
			}
			double cost = cos(angle);
			double sint = sin(angle);
			vector xtemp = cost*xaxis + sint*yaxis;
			yaxis = -sint*xaxis + cost*yaxis;
			xaxis = xtemp;
		}

		// Calculate rotation coefficients from xaxis,yaxis to x,y in the 2D plane
		// a*xaxis + b*yaxis = aa*x + bb*y; dot with x and y to obtain this:
		// aa = (x dot xaxis)*a + (x dot yaxis)*b = c11*a + c12*b
		// bb = (y dot xaxis)*a + (y dot yaxis)*b = c21*a + c22*b
		double c11 = x.dot(xaxis)*s_i[0];
		double c12 = x.dot(yaxis)*s_i[1];
		double c21 = y.dot(xaxis)*s_i[0];
		double c22 = y.dot(yaxis)*s_i[1];

		// This looks like it could be replaced by prevc11 = c11, etc.
		if (delay_initial_face) { // now we have valid x and y
			prevc11 = x.dot(prevxaxis)*tscale[0];
			prevc12 = x.dot(prevyaxis)*tscale[1];
			prevc21 = y.dot(prevxaxis)*tscale[0];
			prevc22 = y.dot(prevyaxis)*tscale[1];
		}

		// Points not on the axle must be rotated around the axle.
		double axlecos = lastA.dot(bisecting_plane_normal); // angle of rotation about axle
		vector xrot;
		if (!axlecos) {
			xrot = x;
		} else {
			xrot = bisecting_plane_normal.cross(y)/axlecos; // make xrot a non-unit vector, in the plane of the joint, to correct for rotation about axle
		}
		xrot *= scene.gcf;
		y *= scene.gcf;

		// update xaxis and yaxis across the joint
		if (icorner == 0) { // special handling due to initial setup of point 0
			if (closed) {
				nextxaxis = xaxis;
				nextyaxis = yaxis;
				xaxis = xaxis.rotate(-theta, y);
				yaxis = yaxis.rotate(-theta, y);
			}
		} else {
			nextxaxis = xaxis.rotate(theta, y);
			nextyaxis = yaxis.rotate(theta, y);
		}

		if (delay_initial_face) { // now we have valid scaled x, y, and xrot
			prevy = y;
			if (smoothed) {
				prevxrot = xrot.rotate(-2*alpha,y)*axlecos;
			} else {
				prevxrot = xrot.rotate(-alpha,y)*axlecos;
			}
		}

		if (icorner == 0 && !zerodepth) {
			if (!closed) {
				delay_initial_face = true; // delay rendering initial face until we have a good normal
				nextxaxis = xaxis;
				nextyaxis = yaxis;
			}

		} else {

			if ((startcorner >= icorner && startcorner <= corner) && (zerodepth || ((!delay_initial_face) && show_start))) {
				// Use pstrips to paint both sides of the first surface
				vector icolor = current_color;
				if (startcorner == 0) icolor = initial_face_color;
				rendered_initial_face = true;
				render_end(-bisecting_plane_normal, current, c11, c12, c21, c22, xrot, y, icolor, true,
						faces_pos, faces_normals, faces_colors, make_faces);
			}

			// This complicated test deals with the situation of a closed path but endcorner < lastpoint.
			if (delay_initial_face || (closed && !rendered_initial_face && startcorner == 0 && endcorner < lastpoint)) {
				delay_initial_face = false;
				rendered_initial_face = true;
				if (show_start) {
					// Use pstrips to paint both sides of the first surface
					if (startcorner == 0) {
						render_end(prevxrot.cross(prevy).norm(), prev, prevc11, prevc12, prevc21, prevc22,
								prevxrot, prevy, initial_face_color, true, faces_pos, faces_normals, faces_colors, make_faces);
					} else {
						if (startcorner <= corner) { // if starting at pos 1
							render_end(-lastA.rotate(-2*alpha, y), current, c11, c12, c21, c22,
									xrot, y, current_color, true, faces_pos, faces_normals, faces_colors, make_faces);
						} else if (endcorner <= corner) { // if ending at pos 1
							render_end(-bisecting_plane_normal, current, c11, c12, c21, c22,
									xrot, y, current_color, true, faces_pos, faces_normals, faces_colors, make_faces);
						}
					}
				}
			}

			if ((endcorner >= icorner && endcorner <= corner) && (!zerodepth && show_end)) {
				vector lastnormal = -bisecting_plane_normal;
				if (!closed && (corner == lastpoint) && prevsmoothed) {
					lastnormal = lastnormal.rotate(lastalpha, y);
					xrot = xrot.rotate(lastalpha,y).norm()*scene.gcf;
				}
				// Use pstrips to paint both sides of the last surface
				vector icolor = current_color;
				if (corner == lastpoint) icolor = final_face_color;
				render_end(lastnormal, current, c11, c12, c21, c22,
						xrot, y, icolor, false, faces_pos, faces_normals, faces_colors, make_faces);
			}

			if (corner > startcorner && corner <= endcorner) {

				if (!make_faces) {
					glVertexPointer(3, GL_DOUBLE, 0, &tris[0]);
					glNormalPointer( GL_DOUBLE, 0, &normals[0]);
					glColorPointer(3, GL_DOUBLE, 0, &tcolors[0]);
				}

				//vector color_old = prev_color; // color at previous location along the curve
				//vector color_new = current_color; // color at current location along the curve
				double v0x, v0y, v1x, v1y, prevv0x, prevv0y, prevv1x, prevv1y;
				// The following nested for loops is (necessarily) the same as that used to build the normals2D array.
				for (size_t c=0, nbase=0; c < ncontours; c++) {
					size_t nd = 2*pcontours[2*c+2]; // number of doubles in this contour
					size_t base = 2*pcontours[2*c+3]; // initial (x,y) = (contour[base], contour[base+1])
					size_t b0, b1, b2, b3;
					// Triangle order is
					//    previous v0, current v1, current v0, previous v1, current v1, previous v0.
					// Render front and back of each triangle.
					for (size_t pt=0, i=0; pt<nd; pt+=2, i+=6, nbase+=4) {
						if (pt == nd-2 && !shape_closed) break;
						// Use modulo arithmetic here because last point is the first point, going around the sides of the extrusion
						b0 = base+pt;
						b1 = b0+1;
						b2 = base+((pt+2)%nd);
						b3 = base+((pt+3)%nd);
						prevv0x = prevc11*contours[b0] + prevc12*contours[b1];
						prevv0y = prevc21*contours[b0] + prevc22*contours[b1];
						prevv1x = prevc11*contours[b2] + prevc12*contours[b3];
						prevv1y = prevc21*contours[b2] + prevc22*contours[b3];
						v0x =     c11*contours[b0]     + c12*contours[b1];
						v0y =     c21*contours[b0]     + c22*contours[b1];
						v1x =     c11*contours[b2]     + c12*contours[b3];
						v1y =     c21*contours[b2]     + c22*contours[b3];

						tris[i  ] = prev    + prevxrot*prevv0x + prevy*prevv0y;
						tris[i+1] = prev    + prevxrot*prevv1x + prevy*prevv1y;
						tris[i+2] = current +     xrot*v0x     + y*v0y;
						tris[i+3] = tris[i+1];
						tris[i+4] = current +     xrot*v1x     + y*v1y;
						tris[i+5] = tris[i+2];

						tcolors[i  ] = prev_color;
						tcolors[i+1] = prev_color;
						tcolors[i+2] = current_color;
						tcolors[i+3] = tcolors[i+1];
						tcolors[i+4] = current_color;
						tcolors[i+5] = tcolors[i+2];

						normals[i  ] = smoothing(     s_i[1]*xaxis*normals2D[nbase  ] +      s_i[0]*yaxis*normals2D[nbase+1],
												 s_i[-2]*prevxaxis*normals2D[nbase  ] + s_i[-3]*prevyaxis*normals2D[nbase+1]);

						normals[i+1] = smoothing(     s_i[1]*xaxis*normals2D[nbase+2] +      s_i[0]*yaxis*normals2D[nbase+3],
												 s_i[-2]*prevxaxis*normals2D[nbase+2] + s_i[-3]*prevyaxis*normals2D[nbase+3]);

						normals[i+2] = smoothing(     s_i[1]*xaxis*normals2D[nbase  ] +      s_i[0]*yaxis*normals2D[nbase+1],
												 s_i[-2]*nextxaxis*normals2D[nbase  ] + s_i[-3]*nextyaxis*normals2D[nbase+1]);

						normals[i+3] = normals[i+1]; // i+1 and i+3 are the same location

						normals[i+4] = smoothing(     s_i[1]*xaxis*normals2D[nbase+2] +      s_i[0]*yaxis*normals2D[nbase+3],
												 s_i[-2]*nextxaxis*normals2D[nbase+2] + s_i[-3]*nextyaxis*normals2D[nbase+3]);

						normals[i+5] = normals[i+2]; // i+2 and i+5 are the same location

						if (!make_faces && twosided) {
							// vertices, normals, and colors for other side
							tris[3*nd+i+1] = tris[i  ];
							tris[3*nd+i  ] = tris[i+1];
							tris[3*nd+i+2] = tris[i+2];
							tris[3*nd+i+3] = tris[i+3];
							tris[3*nd+i+5] = tris[i+4];
							tris[3*nd+i+4] = tris[i+5];

							normals[3*nd+i+1] = -normals[i  ];
							normals[3*nd+i  ] = -normals[i+1];
							normals[3*nd+i+2] = -normals[i+2];
							normals[3*nd+i+3] = -normals[i+3];
							normals[3*nd+i+5] = -normals[i+4];
							normals[3*nd+i+4] = -normals[i+5];

							tcolors[3*nd+i+1] = tcolors[i  ];
							tcolors[3*nd+i  ] = tcolors[i+1];
							tcolors[3*nd+i+2] = tcolors[i+2];
							tcolors[3*nd+i+3] = tcolors[i+3];
							tcolors[3*nd+i+5] = tcolors[i+4];
							tcolors[3*nd+i+4] = tcolors[i+5];
						}

						if (make_faces) {
							faces_pos.insert(faces_pos.end(), tris.begin()+i, tris.begin()+i+6);
							faces_normals.insert(faces_normals.end(), normals.begin()+i, normals.begin()+i+6);
							faces_colors.insert(faces_colors.end(), tcolors.begin()+i, tcolors.begin()+i+6);
						}
					}

					if (!make_faces) {
						// nd doubles, nd/2 vertices, 2 triangles per vertex,
						//    3 points per triangle, 2 sides, so 6*nd vertices per extrusion segment
						if (twosided) {
							glDrawArrays(GL_TRIANGLES, 0, 6*nd);
						} else {
							glDrawArrays(GL_TRIANGLES, 0, 3*nd);
						}
					}
				}
			}
		}
		prevx = x;
		prevy = y;
		prevxrot = xrot;
		prevc11 = c11;
		prevc12 = c12;
		prevc21 = c21;
		prevc22 = c22;
		prevxaxis = xaxis;
		prevyaxis = yaxis;
		xaxis = nextxaxis;
		yaxis = nextyaxis;
		lastA = A;
		lastalpha = alpha;
		prev = current;
		prev_color = vector(c_i[0], c_i[1], c_i[2]);
		prevsmoothed = smoothed;
	}
}

void
extrusion::outer_render(view& v ) {
	arrayprim::outer_render(v);
}

void
extrusion::get_material_matrix( const view& v, tmatrix& out ) {
	//if (degenerate()) return;

	// TODO: note this code is identical to faces::get_material_matrix, except for considering radius

	// TODO: Add some caching for extent with grow_extent etc
	vector min_extent, max_extent;
	const double* pos_i = pos.data();
	const double* pos_end = pos.end();
	min_extent = max_extent = vector( pos_i ); pos_i += 3;
	while (pos_i < pos_end)
		for(int j=0; j<3; j++) {
			if (*pos_i < min_extent[j]) min_extent[j] = *pos_i;
			else if (*pos_i > max_extent[j]) max_extent[j] = *pos_i;
			pos_i++;
		}

	//min_extent -= vector(radius,radius,radius);
	//max_extent += vector(radius,radius,radius);

	out.translate( vector(.5,.5,.5) );
	out.scale( vector(1,1,1) * (.999 / (v.gcf * std::max(max_extent.x-min_extent.x, std::max(max_extent.y-min_extent.y, max_extent.z-min_extent.z)))) );
	out.translate( -.5 * v.gcf * (min_extent + max_extent) );
}

} } // !namespace cvisual::python
