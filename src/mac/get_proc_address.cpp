#include "display_kernel.hpp"
// Mac version of getProcAddress()

#include <dlfcn.h>

namespace cvisual {

display_kernel::EXTENSION_FUNCTION
display_kernel::getProcAddress(const char* name) {
	void *lib = dlopen( (const char *)0L, RTLD_LAZY | RTLD_GLOBAL );
	void *sym = dlsym( lib, name );
	dlclose( lib );
	return (EXTENSION_FUNCTION)sym;
}

} // namespace cvisual
