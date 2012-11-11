#include "material.hpp"

namespace cvisual {

material::material() : translucent(false) {}

void
material::set_textures( std::vector< boost::shared_ptr< texture > > tex ) {
	textures = tex;
}

std::vector< boost::shared_ptr< texture > >
material::get_textures() {
	return textures;
}

void
material::set_shader( const std::string& source ) {
	if (source.size())
		shader.reset( new shader_program( source ) );
	else
		shader.reset( NULL );
}

std::string
material::get_shader() {
	if (shader)
		return shader->get_source();
	else
		return std::string();
}

bool
material::get_translucent() {
	return translucent;
}

void
material::set_translucent( bool t ) {
	translucent = t;
}

apply_material::apply_material( const view& v, material* m, tmatrix& model_material )
 : v(v), sp( v, m ? m->shader.get() : NULL )
{
	if (!m || !sp.ok()) return;
	char texa[] = "tex0";
	for(size_t t=0; t<m->textures.size(); t++) {
		if (t && v.glext.ARB_multitexture)
			v.glext.glActiveTexture(GL_TEXTURE0 + t);
		m->textures[t]->gl_activate(v);

		if (m->shader && v.glext.ARB_shader_objects) {
			texa[3] = '0'+t;
			v.glext.glUniform1iARB( m->shader->get_uniform_location( v, texa ), t );
		}
		if (!v.glext.ARB_multitexture) break;
	}

	// For compatibility, set the texture unit back
	if (m->textures.size() > 1 && v.glext.ARB_multitexture)
		v.glext.glActiveTexture(GL_TEXTURE0);

	int loc;
	if ( (loc = m->shader->get_uniform_location( v, "model_material" )) >= 0 ) {
		m->shader->set_uniform_matrix( v, loc, model_material );
	}
	if ( (loc = m->shader->get_uniform_location( v, "light_count" )) >= 0 )
		v.glext.glUniform1iARB( loc, v.light_count[0] );

	if ( (loc = m->shader->get_uniform_location( v, "light_pos" )) >= 0 && v.light_count[0])
		v.glext.glUniform4fvARB( loc, v.light_count[0], &v.light_pos[0] );

	if ( (loc = m->shader->get_uniform_location( v, "light_color" )) >= 0 && v.light_count[0] )
		v.glext.glUniform4fvARB( loc, v.light_count[0], &v.light_color[0] );
}

apply_material::~apply_material() {
}

} // namespace cvisual
