#include "light.hpp"

namespace cvisual {

void light::render_lights( view& v ) {
	++v.light_count[0];

	vertex p = get_vertex( v.gcf );
	for(int d=0; d<4; d++) v.light_pos.push_back(p[d]);
	for(int d=0; d<3; d++) v.light_color.push_back(color[d]);
	v.light_color.push_back( 1.0 );
}

} // namespace cvisual
