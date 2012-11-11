#ifndef VPYTHON_PYTHON_extrusion_HPP
#define VPYTHON_PYTHON_extrusion_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "util/displaylist.hpp"
#include "python/num_util.hpp"
#include "python/arrayprim.hpp"

namespace cvisual { namespace python {

using boost::python::list;
using boost::python::numeric::array;

class extrusion : public arrayprim_color
{
 protected:
	// The pos and color arrays are always overallocated to make appends
	// faster.  Whenever they are read from Python, we return a slice into the
	// array that starts at its beginning and runs up to the last used position
	// in the array.  This is similar to many implementations of std::vector<>.

	vector up; // Sets initial orientation of the 2D cross section

	bool show_start_face, show_end_face; // if false, don't render the face

	vector first_normal, last_normal;

	double initial_twist; // easier to use than up; note that twist[0] has no effect

	int start, end; // display from start to end (slice notation; -1 is last point)
	size_t startcorner, endcorner; // after gl_render selects points

	double smooth; // smoothing of normals; default is 0.95 (cosine of 18 degrees)

	bool twosided; // default is true (except for faces_render)

	// scale is an array of scale and twist information for the extrusion segments.
	// Each triple is < scalex, scaley, twist >
	arrayprim_array<double> scale;

	virtual void set_length(size_t);

	bool antialias;

	// Returns true if the object is single-colored.
	bool monochrome(double* tcolor, size_t pcount);

	virtual void outer_render(view&);
	virtual void gl_render(view&);
	virtual vector get_center() const;
	virtual void gl_pick_render(view&);
	void get_material_matrix( const view& v, tmatrix& out );
	virtual void grow_extent( extent&);

	// Returns true if the object is degenerate and should not be rendered.
 	bool degenerate() const;

 public:
	extrusion();

	// Add another vertex, up, and color to the extrusion.
	void append_rgb( const vector&, const vector&, float red=-1, float green=-1, float blue=-1);
	void append( const vector&, const vector&, const rgb& );
	void append( const vector&, const vector& );
	void append( const vector& );

	void set_up(const vector&);
	shared_vector& get_up();

	vector get_first_normal();
	void set_first_normal(const vector&);
	vector get_last_normal();
	void set_last_normal(const vector&);

	void set_show_start_face(const bool);
	bool get_show_start_face();
	void set_show_end_face(const bool);
	bool get_show_end_face();

	void set_initial_twist(const double);
	double get_initial_twist();

	void set_start(const int);
	int get_start();
	void set_end(const int);
	int get_end();

	void set_smooth(const double);
	double get_smooth();

	void set_twosided(const bool); // default is true; faces_render is always one-sided
	bool get_twosided();

	void set_twist( const double_array& twist);
	void set_twist_d( const double twist);
	boost::python::object get_twist();

	void set_scale( const double_array& scale);
	void set_scale_d( const double scale);
	void set_xscale( const double_array& scale);
	void set_yscale( const double_array& scale);
	void set_xscale_d( const double scale);
	void set_yscale_d( const double scale);
	boost::python::object get_scale();

	boost::python::object _faces_render();

	void set_contours( const array&, const array&, const array&, const array& );

	// There were unsolvable problems with rotate. See comments with intrude routine.
	//void rotate( double angle, const vector& _axis, const vector& origin);

	void appendpos_retain(const vector&, const int); // position and retain
	void appendpos_color_retain(const vector& n_pos, const double_array& n_color, const int retain);
	void appendpos_rgb_retain(const vector& n_pos,
			const double red, const double green, const double blue, const int retain);

	inline bool get_antialias( void) { return antialias; }

	void set_antialias( bool);

 private:
	bool adjust_colors( const view& scene, double* tcolor, size_t pcount);
	void extrude(const view& scene,
			std::vector<vector>& faces_pos,
			std::vector<vector>& faces_normals,
			std::vector<vector>& faces_colors, bool make_faces);
	void render_end(const vector V, const vector current,
			const double c11, const double c12, const double c21, const double c22,
			const vector xrot, const vector y, const vector current_color, bool show_first,
			std::vector<vector>& faces_pos,
			std::vector<vector>& faces_normals,
			std::vector<vector>& faces_colors, bool make_faces);

	vector smoothing(const vector& a, const vector& b);

	vector calculate_normal(const vector prev, const vector current, const vector next);

	// contours are flattened N*2 arrays of points describing the 2D surface, one after another.
	// pcontours[0] is (number of contours, closed), where closed=1 if closed contour, 0 if not
	// pcontours[2*i+2] is (length of ith contour, starting location of ith contour in contours).
	// strips are flattened N*2 arrays of points describing strips that span the "solid" part of the 2D surface.
	// pstrips[0] is (number of strips, closed)
	// pstrips[2*i] is (length of ith strip, starting location of ith strip in strips).
	std::vector<npy_float64> contours, strips;
	std::vector<npy_int32> pcontours, pstrips;
	std::vector<double> normals2D; // [(nx0,ny0), (nx1,ny1)], [(nx1,ny1), (nx2,ny2)], etc. normals for 2D shape
	bool shape_closed; // 1 if closed shape contour, 0 if not

	vector center; // center of extrusion (seems like it is not used)
	double maxextent; // max scaled distance from curve
	double shape_xmax, shape_ymax; // biggest distances from curve to edges of shape (to calculate max extent)
	size_t maxcontour; // number of vertices in largest contour
};

} } // !namespace cvisual::python

#endif // !VPYTHON_PYTHON_extrusion_HPP
