// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "util/icososphere.hpp"
#include "wrap_gl.hpp"
#include "util/gl_enable.hpp"
#include <cmath>

// sphmodel.h - Deep magic to generate sphere models
//   I generate spheres by subdividing an icosahedron, a Platonic
//     solid with 12 vertices and 20 sides, each an equilateral
//     triangle.  Each triangle is subdivided into four, yielding
//     a figure with 80 sides, 320 sides, and so on.  The complexity
//     is due to the fact that an 80-face "sphere" should have 42
//     vertices, but a naiive algorithm will generate many more
//     (multiple vertices at the same spot).  I avoid this with lots
//     of unmaintainable pointer goop.  If you have to modify this
//     file significantly, just start over.  I warned you! 
//   --David Scherer

#include <cstring>
#include <cassert>

namespace cvisual { namespace /* unnamed */ {

const float SX = .525731112119133606F;
const float SZ = .850650808352039932F;

/* icosahedron model copied from the OpenGL programming guide */
const float vdata[12][3] = {
	{-SX, 0.0, SZ}, {SX, 0.0, SZ}, {-SX, 0.0,-SZ}, {SX, 0.0, -SZ},
	{0.0, SZ, SX}, {0.0, SZ, -SX}, {0.0, -SZ, SX}, {0.0, -SZ, -SX},
	{SZ, SX, 0.0}, {-SZ, SX, 0.0}, {SZ, -SX, 0.0}, {-SZ, -SX, 0.0}
};

const int tindices[20][3] = {
	{0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
	{8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
	{7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
	{6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11}
};

const int edges[30][2] = {
/*  0 */ {0,1}, {0,4}, {0,6}, {0,9}, {0,11},
/*  5 */ {1,4}, {1,6}, {1,8}, {1,10},
/*  9 */ {2,3}, {2,5}, {2,7}, {2,9}, {2,11},
/* 14 */ {3,5}, {3,7}, {3,8}, {3,10},
/* 18 */ {4,5}, {4,8}, {4,9},
/* 21 */ {5,8}, {5,9},
/* 23 */ {6,7}, {6,10}, {6,11},
/* 26 */ {7,10}, {7,11},
/* 28 */ {8,10},
/* 29 */ {9,11}
};

inline void
normalize(float v[3])
{
    float n = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    float m = 1.0f / std::sqrt( n );

    v[0] *= m;
    v[1] *= m;
    v[2] *= m;
}

inline float*
avgptr( float* a, float* b)
{
	return (float*)( long(a) + ((long(b)-long(a))>>1) );
}

} // !namespace anonymous

icososphere::icososphere( int depth)
{
	int span = (1<<depth);
	int triangles = 20 << (depth<<1);
	int vertices = triangles/2 + 2;

	nverts = 12 + 30*(span-1);
	ni = 0;
	verts.reset( new float[ 3*vertices ]);
	indices.reset( new int[ 3*triangles ]);
	// std::cout << "newed " << ((3*vertices*sizeof(float) + 3*triangles*sizeof(int)) >> 10) << " Kbytes of geometry memory\n";
	std::memset(verts.get(), 0, sizeof(float)*3*vertices);
	std::memcpy(verts.get(), vdata, sizeof(vdata));

	for(int t=0; t<20; t++) {
		float *S[3], *E[3];
		for(int e=0; e<3; e++) {
			int which;
			for(which = 0; which < 30; which++) {
				int f = (e+1)%3;
				if (edges[which][0] == tindices[t][e]
					&& edges[which][1] == tindices[t][f]) {
					S[e] = verts.get() + 3*( 12 + which * (span-1) - 1);
					E[e] = S[e] + span*3;
					break;
				}
				else if (edges[which][1] == tindices[t][e]
					&& edges[which][0] == tindices[t][f]) {
					E[e] = verts.get() + 3*( 12 + which * (span-1) - 1);
					S[e] = E[e] + span*3;
					break;
				}
			}
		}
		subdivide( span, 
            verts.get()+3*tindices[t][0], 
            verts.get()+3*tindices[t][1], 
            verts.get()+3*tindices[t][2],
			S[0],S[1],S[2], E[2],E[0],E[1]);
	}
}

void
icososphere::subdivide( int span, float* v1, float* v2, float *v3
	, float* s1, float* s2, float* s3
	, float* e1, float* e2, float* e3 )
{
	if (span > 1) {
		int span2 = span>>1;

		float *v12 = avgptr(s1,e2);
		float *v23 = avgptr(s2,e3);
		float *v31 = avgptr(s3,e1);
		float *s12 = newe(span2);
		float *e31 = s12 + span2*3;
		float *s23 = newe(span2);
		float *e12 = s23 + span2*3;
		float *s31 = newe(span2);
		float *e23 = s31 + span2*3;

		for(int i=0;i<3;i++) {
			v12[i] = v1[i] + v2[i];
			v23[i] = v2[i] + v3[i];
			v31[i] = v3[i] + v1[i];
		}

		normalize(v12);
		normalize(v23);
		normalize(v31);

		subdivide( span2, v1, v12, v31
			, s1, s12, v31
			, e1, v12, e31);
		subdivide( span2, v2, v23, v12
			, s2, s23, v12
			, e2, v23, e12 );
		subdivide( span2, v3, v31, v23
			, s3, s31, v23
			, e3, v31, e23 );
		subdivide( span2, v12, v23, v31
			, e12, e23, e31
			, s12, s23, s31 );
	}
	else {
		indices[ni++] = (v1-verts.get())/3;
		indices[ni++] = (v2-verts.get())/3;
		indices[ni++] = (v3-verts.get())/3;
	}
}

void 
icososphere::gl_render()
{
	gl_enable_client vertexes( GL_VERTEX_ARRAY);
	gl_enable_client normals( GL_NORMAL_ARRAY);
    glVertexPointer( 3, GL_FLOAT, 3*sizeof(float), verts.get());
    glNormalPointer( GL_FLOAT, 3*sizeof(float), verts.get());
    glDrawElements( GL_TRIANGLES, ni, GL_UNSIGNED_INT, indices.get());
}

} // !namespace cvisual
