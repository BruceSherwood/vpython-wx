#ifndef VPYTHON_PYRAMID_HPP
#define VPYTHON_PYRAMID_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "rectangular.hpp"
#include "util/displaylist.hpp"

#include <boost/scoped_ptr.hpp>

namespace cvisual {

using boost::scoped_ptr;

class pyramid : public rectangular
{
 private:
	static displaylist model;
	static void init_model(view& scene);
	friend class arrow;
	
 protected:
	virtual void gl_pick_render( view&);
	virtual void gl_render( view&);
	virtual void grow_extent( extent&);
	virtual vector get_center() const;
	virtual void get_material_matrix( const view&, tmatrix& out );

	PRIMITIVE_TYPEINFO_DECL;
};

} // !namespace cvisual

#endif // !defined VPYTHON_PYRAMID_HPP
