#ifndef VPYTHON_UTIL_ICOSOSPHERE_HPP
#define VPYTHON_UTIL_ICOSOSPHERE_HPP

#include <boost/shared_array.hpp>

namespace cvisual {

class icososphere
{
    boost::shared_array<float> verts;
    boost::shared_array<int> indices;
    int nverts;
    int ni;

    inline float*
    newe( int span)
    {
        float* e = verts.get() + 3*(nverts - 1);  // one before the beginning
        nverts += span-1;
        return e;
    }

    void subdivide( int span, float* v1, float* v2, float *v3, 
        float* s1, float* s2, float* s3,
        float* e1, float* e2, float* e3 );
 public:
	icososphere( int depth);
	void gl_render();
};

} // !namespace cvisual
#endif // !defined VPYTHON_UTIL_ICOSOSPHERE_HPP
