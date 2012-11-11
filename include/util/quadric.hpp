#ifndef VPYTHON_UTIL_QUADRIC_HPP
#define VPYTHON_UTIL_QUADRIC_HPP

// Copyright (c) 2004 by Jonathan Brandmeyer and others
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

class GLUquadric;

namespace cvisual {

/** A thin wrapper around GLU quadric objects.  This may be used as a factory
	to render some predefined quadrics.
*/
class quadric
{
 private:
	GLUquadric* q; ///< The GLU resource being managed.
 
 public:
	enum drawing_style
	{ 
		POINT,
		LINE,
		FILL,
		SILHOUETTE		
	};
	
	enum normal_style
	{
		NONE,
		FLAT,
		SMOOTH		
	};
	
	enum orientation
	{
		OUTSIDE,
		INSIDE
	};
	
	/** Create a new quadric object with smooth normals, filled drawing style,
	 	outside orientation, and no texture coordinates.
	*/
	quadric();
	// Free up any resources that GLU required for the object.
	~quadric();
	
	/** Set the drawing style to be used for subsequently rendered objects. */
	void set_draw_style( drawing_style);
	/** Set the style of generated normal vectors used for subsequently rendered 
		objects. 
	*/
	void set_normal_style( normal_style);
	/** Set the direction of teh normal vectors used for rendered objects. */
	void set_orientation( orientation);
	
	/** Draw a sphere centered at the origin, with the N pole pointing along the
		y axis.
		radius The radius of the generated sphere.
		slices The number of subdivisions around the y axis
		          (similar to lines of longitude).
	    stacks The number of subdivisions along the y axis
		          (similar to lines of latitude).
	*/
	void render_sphere( double radius, int slices, int stacks);
	
	/** Draw a cylinder with these properties.  The cylinder's base is centered
		at the origin, pointing along the +x axis.  Only the outer curve
		of the cylinder is rendered, not the ends.
		base_radius The radius of the base end.
		top_radius The radius of the top end.
		height The distance along +x between the top and bottom.
		slices The number of subdivisions around the x axis.
	    stacks The number of subdivisions along the x axis.
	*/
	void render_cylinder( double base_radius, double top_radius, double height,
		int slices, int stacks);
	
	/** Draw a cylinder with constant radius, as above. */
	void render_cylinder( double radius, double height, int slices, int stacks);
	
	/** Draw a flat filled disk with these properties.  The disk is centered on 
		the	origin, rendered in the yz plane.
		@param radius The outer radius of the disk.
		@param slices The number of radial slices to subdivide the disk into.
		@param rings The number of circumferential subdivisions for the disk.
		@param rotation +1 for right end of cylinder, -1 for left end (or base of cone)
	*/
	void render_disk( double radius, int slices, int rings, GLfloat rotation);
};

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_QUADRIC_HPP
