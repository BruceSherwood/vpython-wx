#ifndef VPYTHON_CONE_HPP
#define VPYTHON_CONE_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "axial.hpp"

namespace cvisual {

class cone : public axial
{
 private:
	static void init_model(view& scene);
	bool degenerate();
	
 public:
	cone();
	
	void set_length( double l);
	double get_length();
	
 protected:
	virtual void gl_pick_render(view&);
	virtual void gl_render(view&);
	virtual void grow_extent( extent&);
	virtual vector get_center() const;
	PRIMITIVE_TYPEINFO_DECL;
};

} // !namespace cvisual

#endif // !defined VPYTHON_CONE_HPP
