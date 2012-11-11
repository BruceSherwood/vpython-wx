#ifndef VPYTHON_ELLIPSOID_HPP
#define VPYTHON_ELLIPSOID_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "sphere.hpp"

namespace cvisual {

class ellipsoid : public sphere
{
 private:
	double height;
	double width;
	
 public:
	ellipsoid();
	
	void set_length( double l);
	double get_length();
	
	void set_height( double h);
	double get_height();
	
	void set_width( double w);
	double get_width();
	
	vector get_size();
	void set_size( const vector&);	
	
 protected:
	virtual vector get_scale();
	virtual void grow_extent( extent&);
	virtual bool degenerate();
	PRIMITIVE_TYPEINFO_DECL;
};

} // !namespace cvisual

#endif // !defined VPYTHON_ELLIPSOID_HPp
