#ifndef VPYTHON_PYTHON_CONVEX_HPP
#define VPYTHON_PYTHON_CONVEX_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "util/sorted_model.hpp"
#include "python/arrayprim.hpp"

#include <vector>

namespace cvisual { namespace python {

class convex : public arrayprim
{
 private:
	struct face : triangle
	{
		double d;
		inline face( const vector& v1, const vector& v2, const vector& v3)
			: triangle( v1, v2, v3), d( normal.dot(corner[0]))
		{
		}
		
		inline bool visible_from( const vector& p)
		{ return normal.dot(p) > d; }
	};
	
	struct edge
	{
		vector v[2];
		inline edge( vector a, vector b)
		{ v[0]=a; v[1]=b; }

		inline bool
		operator==( const edge& b) const
		{
			// There are two cases where a pair of edges are equal, the first is
			// occurs when the endpoints are both the same, while the other occurs 
			// when the edge have the same endpoints in opposite directions.  
			// Since the first case never happens when we construct the hull, we
			// only test for the second case here.
			return (v[0] == b.v[1] && v[1] == b.v[0]);
		}
	};
	
	struct jitter_table
	{
		enum { mask = 1023 };
		enum { count = mask+1 };
		double v[count];

		jitter_table()
		{
			for(int i=0; i<count; i++)
				v[i] = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2 * 1e-6;
		}
	}; 
	static jitter_table jitter;  // Use default construction for initialization.
	
	long last_checksum;
	long checksum() const;
	bool degenerate() const;
	
	// Hull construction routines.
	void recalc();
	void add_point( size_t, vector);
	std::vector<face> hull;
	vector min_extent, max_extent;
	
 public:
	convex();

	void set_color( const rgb&);
	rgb get_color();
	
 protected:
	virtual void gl_render(view&);
	virtual vector get_center() const;
	virtual void gl_pick_render(view&);
	virtual void grow_extent( extent&);
	virtual void get_material_matrix( const view&, tmatrix& out );
};

} } // !namespace cvisual::python

#endif // !defined VPYTHON_PYTHON_CONVEX_HPP
