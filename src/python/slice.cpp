#include "python/slice.hpp"

// Copyright (c) 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.
using namespace boost::python;

namespace cvisual { namespace python {

slice::slice()
	: object( detail::new_reference( 
		PySlice_New( NULL, NULL, NULL)))
{
}

object
slice::start()
{
	return object( detail::borrowed_reference( ((PySliceObject*)this->ptr())->start));
}

object
slice::stop()
{
	return object( detail::borrowed_reference( ((PySliceObject*)this->ptr())->stop));
}

object
slice::step()
{
	return object( detail::borrowed_reference( ((PySliceObject*)this->ptr())->step));
}

} } // !namespace cvisual::python
