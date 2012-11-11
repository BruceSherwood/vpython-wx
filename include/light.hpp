#ifndef VPYTHON_LIGHT_HPP
#define VPYTHON_LIGHT_HPP
#pragma once

// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/tmatrix.hpp"
#include "util/rgba.hpp"
#include "renderable.hpp"

namespace cvisual {

class light : public renderable
{
	protected:
		rgb color;

		virtual vertex get_vertex(double gcf) = 0;

	public:
		virtual rgb get_color() { return color; }
		virtual void set_color( const rgb& r ) { color = r; }

		// renderable protocol
		virtual void outer_render( const view& ) {}
		virtual vector get_center() const { return vector(); }
		virtual void set_material( shared_ptr<class material> ) { throw std::invalid_argument("light object does not have a material."); }
		virtual shared_ptr<class material> get_material() { throw std::invalid_argument("light object does not have a material."); }
		virtual bool is_light() { return true; }
		virtual void render_lights( view& );
};

class local_light : public light {
	protected:
		vector position;
		virtual vertex get_vertex(double gcf) { return vertex( position*gcf, 1.0 ); }
	public:
		virtual const vector& get_pos() { return position; }
		virtual void set_pos(const vector& v) { position = v; }
};

class distant_light : public light {
	protected:
		vector direction;
		virtual vertex get_vertex(double gcf) { return vertex( direction, 0.0 ); }
	public:
		virtual const vector& get_direction() { return direction; }
		virtual void set_direction(const vector& v) { direction = v.norm(); }
};

} // !namespace cvisual

#endif
