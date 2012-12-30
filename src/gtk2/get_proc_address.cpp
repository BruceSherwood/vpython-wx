#include "display_kernel.hpp"
#include <GL/glx.h>

// Linux version of getProcAddress()

namespace cvisual {

display_kernel::EXTENSION_FUNCTION
display_kernel::getProcAddress(const char* name) {
	return (EXTENSION_FUNCTION)glXGetProcAddress((const GLubyte *)name);
}

} // namespace cvisual
