#ifndef VPYTHON_SPHERE_HPP
#define VPYTHON_SPHERE_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "axial.hpp"
#include "util/displaylist.hpp"

namespace cvisual {

/** A simple monochrome sphere. 
 */
class sphere : public axial
{
 private:
	/** The level-of-detail cache.  It is stored for the life of the program, and
		initialized when the first sphere is rendered. At one time there were
		going to be additional entries for the textured case, but that was
		not implemented.
	*/
 	//static displaylist lod_cache[6];
	/// True until the first sphere is rendered, then false.
	static void init_model(view& scene);
 
 public:
	/** Construct a unit sphere at the origin. */
	sphere();
	sphere( const sphere& other);
	virtual ~sphere();
 
 protected:
	/** Renders a simple sphere with the #2 level of detail.  */
	virtual void gl_pick_render( view&);
	/** Renders the sphere.  All of the spheres share the same basic set of 
	 * models, and then use matrix transforms to shape and position them.
	 */
	virtual void gl_render( view&);
	/** Extent reported using extent::add_sphere(). */
	virtual void grow_extent( extent&);
	
	/** Exposed for the benefit of the ellipsoid object, which overrides it. 
	 * The default is to use <radius, radius, radius> for the scale. 
	 */
	virtual vector get_scale();
	/** Returns true if this object should not be drawn.  Conditions are:
	 * zero radius, or visible is false.  (overridden by the ellipsoid class). 
	 */
	virtual bool degenerate();
	
	virtual void get_material_matrix( const view&, tmatrix& out );
	
	PRIMITIVE_TYPEINFO_DECL;
};

} // !namespace cvisual

#endif // !defined VPYTHON_SPHERE_HPP
