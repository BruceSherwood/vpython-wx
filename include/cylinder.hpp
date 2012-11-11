#ifndef VPYTHON_CYLINDER_HPP
#define VPYTHON_CYLINDER_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "axial.hpp"

namespace cvisual {

class cylinder : public axial
{
 private:
	static void init_model(view& scene);
	bool degenerate();
	
 public:
	cylinder();
	cylinder( const cylinder&);
	virtual ~cylinder();
	
	void set_length( double l);
	double get_length();
	
 protected:
	virtual void gl_pick_render( view&);
	virtual void gl_render( view&);
	virtual void grow_extent( extent&);
	virtual vector get_center() const;
	PRIMITIVE_TYPEINFO_DECL;
};

} // !namespace cvisual

#endif // !defined VPYTHON_CYLINDER_HPP
