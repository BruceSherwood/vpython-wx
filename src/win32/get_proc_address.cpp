#include "display_kernel.hpp"
// Windows version of getProcAddress()

namespace cvisual {

display_kernel::EXTENSION_FUNCTION
display_kernel::getProcAddress(const char* name) {
	return (EXTENSION_FUNCTION)::wglGetProcAddress( name );
}

} // namespace cvisual
