#ifndef VPYTHON_RECTANGULAR_HPP
#define VPYTHON_RECTANGULAR_HPP

#include "primitive.hpp"

namespace cvisual {

class rectangular : public primitive
{
 protected:
	double width;
	double height;
	rectangular();
	rectangular( const rectangular& other);

	void apply_transform( const view& );

 public:
	virtual ~rectangular();
	
	void set_length( double l);
	double get_length();
	
	void set_height( double h);
	double get_height();
	
	void set_width( double w);
	double get_width();
	
	vector get_size();
	void set_size( const vector&);	
};	
	
} // !namespace cvisual

#endif // !defined VPYTHON_RECTANGULAR_HPP
