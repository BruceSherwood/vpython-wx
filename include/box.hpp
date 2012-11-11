#ifndef VPYTHON_BOX_HPP
#define VPYTHON_BOX_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "rectangular.hpp"
#include "util/displaylist.hpp"

namespace cvisual {

class box : public rectangular
{
 private:
	// True if the box should not be rendered. 
	bool degenerate();
	//static displaylist model;
	//static void init_model(displaylist& model, bool skip_right_face);
	static void init_model(view& scene, bool skip_right_face);
	friend class arrow;
	
 protected:
	//virtual void gl_pick_render( const view&);
	virtual void gl_pick_render( view&);
	//virtual void gl_render( const view&);
	virtual void gl_render( view&);
	virtual void grow_extent( extent& );
	virtual void get_material_matrix( const view&, tmatrix& out );

	PRIMITIVE_TYPEINFO_DECL;
};

} // !namespace cvisual

#endif // !defined VPYTHON_BOX_HPP
