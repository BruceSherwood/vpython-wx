
#ifndef VPYTHON_WRAP_GL_HPP
#define VPYTHON_WRAP_GL_HPP

// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

//A header file to wrap around GL/gl.h on *nix and Windows.

#if defined(_WIN32)
	#define WIN32_LEAN_AND_MEAN 1
	#define NOMINMAX
	#include <windows.h>
#endif

#if defined(__APPLE__)
	#define GL_GLEXT_LEGACY
	#include <OpenGL/gl.h>
	#include <OpenGL/glu.h>
	#include "GL/glext.h"
#else
	#include <GL/gl.h>
	#include <GL/glext.h>
	#include <GL/glu.h>
#endif

#endif // !defined VPYTHON_WRAP_GL_HPP
