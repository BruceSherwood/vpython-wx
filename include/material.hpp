#ifndef VPYTHON_MATERIAL_HPP
#define VPYTHON_MATERIAL_HPP
#pragma once

#include "util/texture.hpp"
#include "util/shader_program.hpp"

namespace cvisual {

class material {
 public:
	material();

	void set_textures( std::vector< boost::shared_ptr< texture > > );
	std::vector< boost::shared_ptr< texture > > get_textures();
	
	void set_shader( const std::string& );
	std::string get_shader();
	
	void set_translucent( bool );
	bool get_translucent();
	
	shader_program* get_shader_program() { return shader.get(); }

 private:
	friend class apply_material;
	
	std::vector< boost::shared_ptr< texture > > textures;
	boost::scoped_ptr< shader_program > shader;
	bool translucent;
};

class apply_material {
 public:
	apply_material( const view& v, material* m, tmatrix& material_matrix );
	~apply_material();

 private:
	const view& v;
	use_shader_program sp;
};

} // namespace cvisual

#endif
