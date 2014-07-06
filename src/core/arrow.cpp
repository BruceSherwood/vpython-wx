// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "arrow.hpp"
#include "util/errors.hpp"
#include "util/gl_enable.hpp"
#include "box.hpp"
#include "pyramid.hpp"
#include "material.hpp"

namespace cvisual {

bool
arrow::degenerate()
{
	return axis.mag() == 0.0;
}

arrow::arrow()
	: fixedwidth(false), headwidth(0), headlength(0), shaftwidth(0)
{
}

arrow::arrow( const arrow& other)
	: primitive(other), fixedwidth( other.fixedwidth),
	headwidth( other.headwidth), headlength( other.headlength),
	shaftwidth( other.shaftwidth)
{
}

arrow::~arrow()
{
}

void
arrow::set_headwidth( double hw)
{
	headwidth = hw;
}

double
arrow::get_headwidth()
{
	if (headwidth) return headwidth;
	if (shaftwidth) return 2.0*shaftwidth;
	return 0.2*axis.mag();
}

void
arrow::set_headlength( double hl)
{
	headlength = hl;
}

double
arrow::get_headlength()
{
	if (headlength) return headlength;
	if (shaftwidth) return 3.0*shaftwidth;
	return 0.3*axis.mag();
}

void
arrow::set_shaftwidth( double sw)
{
	shaftwidth = sw;
	fixedwidth = true;
}

double
arrow::get_shaftwidth()
{
	if (shaftwidth) return shaftwidth;
	return 0.1*axis.mag();
}

void
arrow::set_fixedwidth( bool fixed)
{
	fixedwidth = fixed;
}

bool
arrow::is_fixedwidth()
{
	return fixedwidth;
}

void
arrow::set_length( double l)
{
	axis = axis.norm() * l;
}

double
arrow::get_length()
{
	return axis.mag();
}

vector
arrow::get_center() const
{
	return (pos + axis)/2.0;
}

void
arrow::gl_pick_render(view& scene)
{
	// TODO: material related stuff in this file really needs cleaning up!
	boost::shared_ptr<material> m;
	m.swap(mat);
	gl_render(scene);
	m.swap(mat);
}

void
//arrow::gl_render( const view& scene)
arrow::gl_render( view& scene)
{
	if (degenerate()) return;

	//init_model();
	init_model(scene);

	color.gl_set(opacity);

	double hl,hw,len,sw;
	effective_geometry( hw, sw, len, hl, 1.0 );

	int model_material_loc = mat && mat->get_shader_program() ? mat->get_shader_program()->get_uniform_location( scene, "model_material" ) : -1;

	// Render the shaft and the head in back to front order (the shaft is in front
	// of the head if axis points away from the camera)
	int shaft = axis.dot( scene.camera - (pos + axis * (1-hl/len)) ) < 0;
	for(int part=0; part<2; part++) {
		gl_matrix_stackguard guard;
		model_world_transform( scene.gcf ).gl_mult();
		if (part == shaft) {
			glScaled( len - hl, sw, sw );
			glTranslated( 0.5, 0, 0 );

			if (model_material_loc >= 0) {  // TODO simplify
				tmatrix model_mat;
				double s = 1.0 / std::max( len, hw );
				model_mat.translate( vector((len-hl)*s*0.5,0.5,0.5) );
				model_mat.scale( vector((len-hl), sw, sw)*s );
				mat->get_shader_program()->set_uniform_matrix( scene, model_material_loc, model_mat );
			}

			scene.box_model.gl_render();
		} else {
			glTranslated( len - hl, 0, 0 );
			glScaled( hl, hw, hw );

			if (model_material_loc >= 0) {  // TODO simplify
				tmatrix model_mat;
				double s = 1.0 / std::max( len, hw );
				model_mat.translate( vector((len-hl)*s,0.5,0.5) );
				model_mat.scale( vector(hl, hw, hw)*s );
				mat->get_shader_program()->set_uniform_matrix( scene, model_material_loc, model_mat );
			}

			scene.pyramid_model.gl_render();
		}
	}
}

void
arrow::grow_extent( extent& world)
{
	if (degenerate())
		return;
	double hl, hw, len, sw;
	effective_geometry( hw, sw, len, hl, 1.0);
	vector x = axis.cross(up).norm() * 0.5;
	vector y = axis.cross(x).norm() * 0.5;
	vector base = pos + axis.norm()*(len-hl);
	for(int i=-1; i<=+1; i+=2)
		for(int j=-1; j<=+1; j+=2) {
			world.add_point( pos + x*(i*sw) + y*(j*sw) );
			world.add_point( base + x*(i*hw) + y*(j*hw) );
		}
	world.add_point( pos + axis);
	world.add_body();
}

void
arrow::get_material_matrix(const view& v, tmatrix& out)
{
	// This work is done in gl_render, for shaft and head separately
}

void arrow::init_model(view& scene)
{
	if (!scene.box_model.compiled()) box::init_model(scene, false);
	if (!scene.pyramid_model.compiled()) pyramid::init_model(scene);
}

PRIMITIVE_TYPEINFO_IMPL(arrow)

void
arrow::effective_geometry(
	double& eff_headwidth, double& eff_shaftwidth, double& eff_length,
	double& eff_headlength, double gcf)
{
	// First calculate the actual geometry based on the specs for headwidth,
	// shaftwidth, shaftlength, and fixedwidth.  This geometry is calculated
	// in world space and multiplied
	static const double min_sw = 0.02; // minimum shaftwidth
	static const double def_sw = 0.1; // default shaftwidth
	static const double def_hw = 2.0; // default headwidth multiplier. (x shaftwidth)
	static const double def_hl = 3.0; // default headlength multiplier. (x shaftwidth)
	// maximum fraction of the total arrow length allocated to the head.
	static const double max_headlength = 0.5;

	eff_length = axis.mag() * gcf;
	if (shaftwidth)
		eff_shaftwidth = shaftwidth * gcf;
	else
		eff_shaftwidth = eff_length * def_sw;

	if (headwidth)
		eff_headwidth = headwidth * gcf;
	else
		eff_headwidth = eff_shaftwidth * def_hw;

	if (headlength)
		eff_headlength = headlength * gcf;
	else
		eff_headlength = eff_shaftwidth * def_hl;

	if (fixedwidth) {
		if (eff_headlength > max_headlength * eff_length)
			eff_headlength = max_headlength * eff_length;
	}
	else {
		if (eff_shaftwidth < eff_length * min_sw) {
			double scale = eff_length * min_sw / eff_shaftwidth;
			eff_shaftwidth = eff_length * min_sw;
			eff_headwidth *= scale;
			eff_headlength *= scale;
		}
		if (eff_headlength > eff_length * max_headlength) {
			double scale = eff_length * max_headlength / eff_headlength;
			eff_headlength = eff_length * max_headlength;
			eff_headwidth *= scale;
			eff_shaftwidth *= scale;
		}
	}
}

} // !namespace cvisual
