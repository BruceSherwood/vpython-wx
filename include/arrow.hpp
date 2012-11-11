#ifndef VPYTHON_ARROW_HPP
#define VPYTHON_ARROW_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "primitive.hpp"
#include "util/displaylist.hpp"

namespace cvisual {

/** A 3D 4-sided arrow, with adjustable head and shaft. **/
class arrow : public primitive
{
 private:
	/** True if the width of the point and shaft should not vary with the length
		of the arrow. 
	 */
	bool fixedwidth;
	
	/** If zero, then use automatic scaling for the width's of the parts of the
		arrow.  If nonzero, they specify proportions for the arrow in world 
		space.
	*/
	double headwidth;
	double headlength;
	double shaftwidth;

	void init_model(view& scene);
	bool degenerate();

	/** Initializes these four variables with the effective geometry for the
		arrow.  The resulting geometry is scaled to view space, but oriented
		and positioned in model space.  The only requred transforms are
		reorientation and translation.
	*/
	void effective_geometry( 
		double& headwidth, double& shaftwidth, double& length, 
		double& headlength, double gcf);
 
 public:
	/** Default arrow.  Pointing along +x, unit length, 
	 */
	arrow();
	arrow( const arrow& other);
	virtual ~arrow();
	
	void set_headwidth( double hw);
	double get_headwidth();
	
	void set_headlength( double hl);
	double get_headlength();
	
	void set_shaftwidth( double sw);
	double get_shaftwidth();
	
	void set_fixedwidth( bool fixed);
	bool is_fixedwidth();
	
	void set_length( double l);
	double get_length();
	
 protected:
	virtual void gl_pick_render(view&);
	virtual void gl_render(view&);

	virtual void grow_extent( extent&);
	virtual vector get_center() const;
	virtual void get_material_matrix(const view&, tmatrix& out);
	
	PRIMITIVE_TYPEINFO_DECL;
};

} // !namespace cvisual

#endif // !defined VPYTHON_ARROW_HPP
