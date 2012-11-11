#ifndef VPYTHON_AXIAL_HPP
#define VPYTHON_AXIAL_HPP

#include "primitive.hpp"

namespace cvisual {

/** A subbase class used to only export 'radius' as a property once to Python. */
class axial : public primitive
{
 protected:
 	/// The radius of whatever body inherits from this class.
	double radius;
	axial();
	axial( const axial& other);
	
 public:
 	virtual ~axial();
 	void set_radius(double r);
 	double get_radius();
	virtual void get_material_matrix( const view&, tmatrix& out );
};

} // !namespace cvisual

#endif // !defined VPYTHON_AXIAL_HPP
