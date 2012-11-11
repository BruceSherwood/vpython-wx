// This alternative to numpy is not currently being used
// but is retained in CVS for possible future use.
// In particular, information about array objects such
// as curve cannot currently be cached because we don't
// know when a numpy pos array has been changed.

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include <boost/python/class.hpp>
#include <boost/python/object.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/init.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/def.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/iterator.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/module.hpp>

#include "python/vector_array.hpp"
#include "python/scalar_array.hpp"

// TODO: Figure out what to do with this...
#define PY_ARRAY_UNIQUE_SYMBOL visual_PyArrayHandle
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <iostream>
#include <stdexcept>

namespace cvisual {namespace python {

		vector_array::vector_array( const boost::python::list& sequence)
		: data( boost::python::extract<int>( sequence.attr("__len__")()))
		{
			iterator i = data.begin();
			for (int s_i = 0; s_i < sequence.attr("__len__")(); ++s_i, ++i) {
				boost::python::extract<vector> v_extractor( sequence[s_i]);
				if (v_extractor.check()) {
					*i = v_extractor();
				}
				else {
					boost::python::object elem = sequence[s_i];
					*i = vector();
					switch (boost::python::extract<int>(elem.attr("__len__")())) {
						case 3:
						i->z = boost::python::extract<double>(elem[2]);
						case 2:
						i->y = boost::python::extract<double>(elem[1]);
						i->x = boost::python::extract<double>(elem[0]);
						default:
						throw std::invalid_argument(
								"Can only construct a vector from a "
								"sequence of 2 or 3 doubles.");
					}
				}
			}
		}

		vector_array::vector_array( boost::python::numeric::array sequence)
		: data( ((PyArrayObject*)sequence.ptr())->dimensions[0])
		{
			const PyArrayObject* seq_ptr = (const PyArrayObject*)sequence.ptr();

			if (!( seq_ptr->nd == 2
							&& seq_ptr->dimensions[1] == 3
							&& seq_ptr->descr->type_num == PyArray_DOUBLE)) {
				throw std::invalid_argument( "Must construct a vector_array from an Nx3 array of type Float64.");
			}

			const double* seq_i = (const double*)seq_ptr->data;
			iterator i = this->begin();
			for (; i != this->end(); ++i, seq_i += 3) {
				*i = vector( seq_i[0], seq_i[1], seq_i[2]);
			}
		}

		void
		vector_array::append( const vector& v)
		{
			data.push_back( v);
		}

		void
		vector_array::append( const vector_array& va)
		{
			data.insert( data.end(), va.data.begin(), va.data.end());
		}

		void
		vector_array::prepend( const vector& v)
		{
			data.push_front( v);
		}

		void
		vector_array::head_clip()
		{
			data.pop_front();
		}

		void
		vector_array::head_crop( int i_)
		{
			if (i_ < 0)
			throw std::invalid_argument( "Cannot crop a negative amount.");
			size_t i = (size_t)i_;
			if (i >= data.size())
			throw std::invalid_argument( "Cannot crop greater than the array's length.");

			iterator begin = data.begin();
			data.erase( begin, begin+i);
		}

		void
		vector_array::tail_clip()
		{
			data.pop_back();
		}

		void
		vector_array::tail_crop( int i_)
		{
			if (i_ < 0)
			throw std::invalid_argument( "Cannot crop a negative amount.");
			size_t i = (size_t)i_;
			if (i >= data.size())
			throw std::invalid_argument( "Cannot crop greater than the array's length.");

			iterator end = data.end();
			data.erase( end-i, end);
		}

		vector_array
		vector_array::operator*( double s) const
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = *i * s;
			}
			return ret;
		}

		// Possibly dangerous, this provides block multiplication elementwise, not to be
		// confused with dot() or cross().
		vector_array
		vector_array::operator*( vector v) const
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				vector v_i = *i;
				*r_i = vector( v_i[0]*v[0], v_i[1]*v[1], v_i[2]*v[2]);
			}
			return ret;
		}

		vector_array
		vector_array::operator*( const scalar_array& s) const
		{
			if (data.size() != s.data.size())
			throw std::out_of_range( "Incompatible vector array multiplication.");

			vector_array ret( data.size());
			scalar_array::const_iterator s_i = s.begin();
			iterator r_i = ret.begin();

			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i, ++s_i) {
				*r_i = *i * *s_i;
			}
			return ret;
		}

		vector_array
		vector_array::operator/( double s) const
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = *i / s;
			}
			return ret;
		}

		vector_array
		vector_array::operator/( const scalar_array& s) const
		{
			if (data.size() != s.data.size())
			throw std::out_of_range( "Incompatible vector array division.");

			vector_array ret( data.size());
			scalar_array::const_iterator s_i = s.begin();
			iterator r_i = ret.begin();

			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i, ++s_i) {
				*r_i = *i / *s_i;
			}
			return ret;
		}

		vector_array
		vector_array::operator-() const
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = -(*i);
			}
			return ret;
		}

		const vector_array&
		vector_array::operator*=( double s)
		{
			for (iterator i = data.begin(); i != data.end(); ++i) {
				*i *= s;
			}
			return *this;
		}

		const vector_array&
		vector_array::operator*=( const scalar_array& s)
		{
			if (data.size() != s.data.size())
			throw std::out_of_range( "Incompatible vector array multiplication.");

			scalar_array::const_iterator s_i = s.begin();
			for (iterator i = data.begin(); i != data.end(); ++i, ++s_i) {
				*i *= *s_i;
			}
			return *this;
		}

		const vector_array&
		vector_array::operator/=( double s)
		{
			for (iterator i = data.begin(); i != data.end(); ++i) {
				*i /= s;
			}
			return *this;
		}

		const vector_array&
		vector_array::operator/=( const scalar_array& s)
		{
			if (data.size() != s.data.size())
			throw std::out_of_range( "Incompatible vector array multiplication.");

			scalar_array::const_iterator s_i = s.begin();
			for (iterator i = data.begin(); i != data.end(); ++i, ++s_i) {
				*i /= *s_i;
			}
			return *this;
		}

		vector_array
		vector_array::operator+( const vector& v) const
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = *i + v;
			}
			return ret;
		}

		vector_array
		vector_array::operator+( const vector_array& v) const
		{
			if (data.size() != v.data.size())
			throw std::out_of_range( "Incompatible vector array addition.");

			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			const_iterator v_i = v.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i, ++v_i) {
				*r_i = *i + *v_i;
			}
			return ret;
		}

		vector_array
		vector_array::operator-( const vector& v) const
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i ) {
				*r_i = *i - v;
			}
			return ret;
		}

		vector_array
		vector_array::operator-( const vector_array& v) const
		{
			if (data.size() != v.data.size())
			throw std::out_of_range( "Incompatible vector array subtraction.");

			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			const_iterator v_i = v.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i, ++v_i) {
				*r_i = *i - *v_i;
			}
			return ret;
		}

		const vector_array&
		vector_array::operator+=( const vector& v)
		{
			for (iterator i = data.begin(); i != data.end(); ++i) {
				*i += v;
			}
			return *this;
		}

		const vector_array&
		vector_array::operator+=( const vector_array& v)
		{
			if (data.size() != v.data.size())
			throw std::out_of_range( "Incompatible vector array addition.");

			const_iterator v_i = v.data.begin();
			for (iterator i = data.begin(); i != data.end(); ++i, ++v_i) {
				*i += *v_i;
			}
			return *this;
		}

		const vector_array&
		vector_array::operator-=( const vector& v)
		{
			for (iterator i = data.begin(); i != data.end(); ++i) {
				*i -= v;
			}
			return *this;
		}

		const vector_array&
		vector_array::operator-=( const vector_array& v)
		{
			if (data.size() != v.data.size())
			throw std::out_of_range( "Incompatible vector array subtraction.");

			const_iterator v_i = v.data.begin();
			for (iterator i = data.begin(); i != data.end(); ++i, ++v_i) {
				*i -= *v_i;
			}
			return *this;

		}

		vector_array
		vector_array::cross( const vector& v)
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->cross(v);
			}
			return ret;
		}

		vector_array
		vector_array::cross( const vector_array& v)
		{
			if (v.data.size() != data.size())
			throw std::out_of_range( "Incompatible vector_array types." );

			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			const_iterator v_i = v.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i, ++v_i) {
				*r_i = i->cross( *v_i);
			}
			return ret;
		}

		vector_array
		vector_array::norm() const
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->norm();
			}
			return ret;
		}

		vector_array
		vector_array::fabs() const
		{
			vector_array ret( data.size());

			iterator r_i = ret.data.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->fabs();
			}
			return ret;
		}

		vector_array
		vector_array::proj( const vector& v)
		{
			vector_array ret( data.size());
			iterator r_i = ret.data.begin();

			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->proj( v);
			}
			return ret;
		}

		vector_array
		vector_array::proj( const vector_array& v)
		{
			if (v.data.size() != data.size())
			throw std::out_of_range( "Incompatible vector_array types." );

			vector_array ret( data.size());
			iterator r_i = ret.data.begin();
			const_iterator v_i = v.data.begin();

			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i, ++v_i) {
				*r_i = i->proj( *v_i);
			}
			return ret;
		}

		scalar_array
		vector_array::mag() const
		{
			scalar_array ret( data.size());
			scalar_array::iterator r_i = ret.begin();
			for ( const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->mag();
			}
			return ret;
		}

		scalar_array
		vector_array::mag2() const
		{
			scalar_array ret( data.size());
			scalar_array::iterator r_i = ret.begin();
			for ( const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->mag2();
			}
			return ret;

		}

		void
		vector_array::rotate( const double& d, vector axis)
		{
			for (iterator i = data.begin(); i != data.end(); ++i) {
				i->rotate( d, axis);
			}
		}

		vector&
		vector_array::py_getitem( int index)
		{
			if (index < 0) {
				// Negative indexes are counted from the end of the array in Python.
				index += data.size();
			}

			return data.at(index);
		}

		void
		vector_array::py_setitem( int index, vector value)
		{
			if (index < 0)
			index += data.size();

			data.at(index) = value;
		}

		scalar_array
		vector_array::dot( const vector& v)
		{
			scalar_array ret( data.size());
			scalar_array::iterator r_i = ret.begin();
			for ( const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->dot( v);
			}
			return ret;
		}

		scalar_array
		vector_array::dot( const vector_array& v)
		{
			if (v.data.size() != data.size())
			throw std::out_of_range( "Incompatible vector_array types." );

			scalar_array ret( data.size());
			scalar_array::iterator r_i = ret.begin();
			const_iterator v_i = v.begin();

			for ( const_iterator i = data.begin(); i != data.end(); ++i, ++r_i, ++v_i) {
				*r_i = i->dot( *v_i);
			}
			return ret;
		}

		scalar_array
		vector_array::comp( const vector& v)
		{
			scalar_array ret( data.size());

			scalar_array::iterator r_i = ret.begin();
			for (iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->comp( v);
			}
			return ret;
		}

		scalar_array
		vector_array::comp( const vector_array& v)
		{
			if ( data.size() != v.data.size() )
			throw std::out_of_range( "Incompatible array scalar projection.");

			scalar_array ret( data.size());

			scalar_array::iterator r_i = ret.begin();
			const_iterator v_i = v.begin();
			for (const_iterator i = data.begin(); i !=data.end(); ++i, ++v_i, ++r_i) {
				*r_i = i->comp( *v_i);
			}
			return ret;
		}

		scalar_array
		vector_array::get_x() const
		{
			scalar_array ret( data.size());
			scalar_array::iterator r_i = ret.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->get_x();
			}
			return ret;
		}

		scalar_array
		vector_array::get_y() const
		{
			scalar_array ret( data.size());
			scalar_array::iterator r_i = ret.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->get_y();
			}
			return ret;
		}

		scalar_array
		vector_array::get_z() const
		{
			scalar_array ret( data.size());
			scalar_array::iterator r_i = ret.begin();
			for (const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = i->get_z();
			}
			return ret;
		}

		void
		vector_array::set_x( const scalar_array& x)
		{
			if (data.empty()) {
				// In case we are building this thing up from single columns
				data = std::deque<vector>( x.size());
			}
			if (x.data.size() != data.size())
			throw std::out_of_range( "Incompatible array assignment.");

			scalar_array::const_iterator x_i = x.begin();
			for (iterator i = data.begin(); i != data.end(); ++i, ++x_i) {
				i->set_x( *x_i);
			}
		}

		void
		vector_array::set_y( const scalar_array& y)
		{
			if (data.empty()) {
				// In case we are building this thing up from single columns
				data = std::deque<vector>( y.size());
			}
			if (y.data.size() != data.size())
			throw std::out_of_range( "Incompatible array assignment.");

			scalar_array::const_iterator y_i = y.begin();
			for (iterator i = data.begin(); i != data.end(); ++i, ++y_i) {
				i->set_y( *y_i);
			}
		}

		void
		vector_array::set_z( const scalar_array& z)
		{
			if (data.empty()) {
				// In case we are building this thing up from single columns
				data = std::deque<vector>( z.size());
			}
			if (z.data.size() != data.size())
			throw std::out_of_range( "Incompatible array assignment.");

			scalar_array::const_iterator z_i = z.begin();
			for (iterator i = data.begin(); i != data.end(); ++i, ++z_i) {
				i->set_z( *z_i);
			}
		}

		void
		vector_array::set_x( const boost::python::list& sequence)
		{
			this->set_x( scalar_array( sequence));
		}

		void
		vector_array::set_y( const boost::python::list& sequence)
		{
			this->set_y( scalar_array( sequence));
		}

		void
		vector_array::set_z( const boost::python::list& sequence)
		{
			this->set_z( scalar_array( sequence));
		}

		vector
		vector_array::sum() const
		{
			vector ret;
			for (const_iterator i = data.begin(); i != data.end(); ++i) {
				ret += *i;
			}
			return ret;
		}

		// Evaluate expression vector - vector_array
		vector_array
		vector_array::lhs_sub( const vector& v) const
		{
			vector_array ret( data.size());
			iterator r_i = ret.begin();
			for ( const_iterator i = data.begin(); i != data.end(); ++i, ++r_i) {
				*r_i = v - *i;
			}
			return ret;
		}

		vector_array
		vector_array::operator>=( const double& s) const
		{
			vector_array ret( data.size());
			iterator r_i = ret.begin();
			for (const_iterator i = data.begin(); i!= data.end(); ++i, ++r_i) {
				*r_i = vector( (i->x >= s) ? 1.0: 0.0
						, (i->y >= s) ? 1.0: 0.0
						, (i->z >= s) ? 1.0: 0.0);
			}
			return ret;
		}

		vector_array
		vector_array::operator>=( const scalar_array& s) const
		{
			if (s.data.size() != data.size())
			throw std::out_of_range( "Incompatible vector_array to scalar_array comparison." );

			vector_array ret( data.size());
			iterator r_i = ret.begin();
			scalar_array::const_iterator s_i = s.begin();
			for (const_iterator i = data.begin(); i!= data.end(); ++i, ++r_i, ++s_i) {
				*r_i = vector( (i->x >= *s_i) ? 1.0: 0.0
						, (i->y >= *s_i) ? 1.0: 0.0
						, (i->z >= *s_i) ? 1.0: 0.0);
			}
			return ret;
		}

		vector_array
		vector_array::operator>=( const vector_array& v) const
		{
			if (v.data.size() != data.size())
			throw std::out_of_range( "Incompatible vector_array to vector_array comparison" );

			vector_array ret( data.size());
			iterator r_i = ret.begin();
			const_iterator v_i = v.begin();
			for (const_iterator i = data.begin(); i!= data.end(); ++i, ++r_i, ++v_i) {
				*r_i = vector( (i->x >= v_i->x) ? 1.0: 0.0
						, (i->y >= v_i->y) ? 1.0: 0.0
						, (i->z >= v_i->z) ? 1.0: 0.0);
			}
			return ret;
		}

		vector_array
		vector_array::operator<=( const double& s) const
		{
			vector_array ret( data.size());
			iterator r_i = ret.begin();
			for (const_iterator i = data.begin(); i!= data.end(); ++i, ++r_i) {
				*r_i = vector( (i->x <= s) ? 1.0: 0.0
						, (i->y <= s) ? 1.0: 0.0
						, (i->z <= s) ? 1.0: 0.0);
			}
			return ret;
		}

		vector_array
		vector_array::operator<=( const scalar_array& s) const
		{
			if (s.data.size() != data.size())
			throw std::out_of_range( "Incompatible vector_array comparison." );

			vector_array ret( data.size());
			iterator r_i = ret.begin();
			scalar_array::const_iterator s_i = s.begin();
			for (const_iterator i = data.begin(); i!= data.end(); ++i, ++r_i, ++s_i) {
				*r_i = vector( (i->x <= *s_i) ? 1.0: 0.0
						, (i->y <= *s_i) ? 1.0: 0.0
						, (i->z <= *s_i) ? 1.0: 0.0);
			}
			return ret;
		}

		vector_array
		vector_array::operator<=( const vector_array& v) const
		{
			if (v.data.size() != data.size())
			throw std::out_of_range( "Incompatible vector_array comparison." );

			vector_array ret( data.size());
			iterator r_i = ret.begin();
			const_iterator v_i = v.begin();
			for (const_iterator i = data.begin(); i!= data.end(); ++i, ++r_i, ++v_i) {
				*r_i = vector( (i->x <= v_i->x) ? 1.0: 0.0
						, (i->y <= v_i->y) ? 1.0: 0.0
						, (i->z <= v_i->z) ? 1.0: 0.0);
			}
			return ret;
		}

		vector_array
		vector_array::operator*( const vector_array& v) const
		{
			if (v.data.size() != data.size())
			throw std::out_of_range( "Incompatible vector_array multiplication." );

			vector_array ret( data.size());
			iterator r_i = ret.begin();
			const_iterator v_i = v.begin();
			for (const_iterator i = data.begin(); i!= data.end(); ++i, ++r_i, ++v_i) {
				*r_i = vector( (i->x * v_i->x)
						, (i->y * v_i->y)
						, (i->z * v_i->z));
			}
			return ret;
		}

		void
		vector_array::set_x( boost::python::numeric::array x)
		{
			this->set_x( scalar_array( x));
		}

		void
		vector_array::set_y( boost::python::numeric::array y)
		{
			this->set_y( scalar_array( y));
		}

		void
		vector_array::set_z( boost::python::numeric::array z)
		{
			this->set_z( scalar_array( z));
		}

		void
		vector_array::set_x( double x)
		{
			this->set_x( scalar_array( size(), x));
		}

		void
		vector_array::set_y( double y)
		{
			this->set_y( scalar_array( size(), y));
		}

		void
		vector_array::set_z( double z)
		{
			this->set_z( scalar_array( size(), z));
		}

		boost::python::handle<PyObject>
		vector_array::as_array() const
		{
			// Make space for the returned array
			int dims[] = {size(), 3};
			boost::python::handle<> ret( PyArray_FromDims( 2, dims, PyArray_DOUBLE));

			// A direct pointer to the PyArrayObject
			PyArrayObject* ret_ptr = (PyArrayObject *)ret.get();

			// Iterable pointers to copy the data.
			double* r_i = (double *) ret_ptr->data;
			const_iterator i = this->begin();
			// Copy the data.
			for (; i != this->end(); ++i, r_i += 3) {
				r_i[0] = i->get_x();
				r_i[1] = i->get_y();
				r_i[2] = i->get_z();
			}
			return ret;
		}

		// This algorithm is failing to recognize all of the collisions...
		boost::python::list
		sphere_collisions( const vector_array& pos, const scalar_array& radius)
		{
			if (pos.size() != radius.size())
			throw std::out_of_range( "Unequal array lengths.");
			assert( pos.size());

			boost::python::list ret;
			vector_array::const_iterator pos_i = pos.begin();
			scalar_array::const_iterator r_i = radius.begin();
			// In spite of using two iterators and a counter, this is a two-pointer
			// search: i, and j.
			for ( int i = 0; pos_i != pos.end(); ++pos_i, ++r_i, ++i) {
				vector_array::const_iterator pos_j = pos_i + 1;
				scalar_array::const_iterator r_j = r_i + 1;
				for ( int j = i+1; pos_j != pos.end(); ++pos_j, ++r_j, ++j) {
					// If the magnitude of the differential distance is smaller
					// than the radius between them, a collision!
					if ((*pos_i - *pos_j).mag() < (*r_i + *r_j)) {
						// A collision
						ret.append( boost::python::make_tuple( i, j));
					}
				}
			}
			return ret;
		}

		// Return a list of integers for those spheres which have collided with
		// the plane specified by a normal vector 'normal' and point on the plane 'origin'
		boost::python::list
		sphere_to_plane_collisions( const vector_array& pos
				, const scalar_array& radius
				, vector normal
				, vector origin)
		{
			boost::python::list ret;
			vector_array::const_iterator p_i = pos.begin();
			scalar_array::const_iterator r_i = radius.begin();
			for( int i = 0; p_i != pos.end(); ++p_i, ++r_i, ++i) {
				// Compute the scalar projection of a vector from the origin to pos
				// to the normal vector of the plane.
				double dist = (*p_i - origin).comp( normal);
				if ( dist < *r_i) {
					ret.append( i);
				}
			}
			return ret;
		}

		/************************ python interface code ******************************/

		namespace {
			BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS( vector_array_rotate, vector_array::rotate, 1, 2)
		}

		void
		wrap_vector_array()
		{
			using namespace boost::python;
			// For the uninitiated (and sane), these are member function pointers.
			// They have the following syntax:
			// return-type (class-type::* pointer-name)(argument-list) = &class-type::member-function;


			vector_array (vector_array::* proj_vector)(const vector&) = &vector_array::proj;
			vector_array (vector_array::* proj_vector_array)( const vector_array&) = &vector_array::proj;
			vector_array (vector_array::* cross_vector)(const vector&) = &vector_array::cross;
			vector_array (vector_array::* cross_vector_array)(const vector_array&) = &vector_array::cross;

			scalar_array (vector_array::* dot_vector)(const vector&) = &vector_array::dot;
			scalar_array (vector_array::* dot_vector_array)(const vector_array&) = &vector_array::dot;
			scalar_array (vector_array::* comp_vector) (const vector&) = &vector_array::comp;
			scalar_array (vector_array::* comp_vector_array)(const vector_array&) = &vector_array::comp;

			void (vector_array::* append_vector)(const vector&) = &vector_array::append;
			void (vector_array::* prepend_vector)(const vector&) = &vector_array::prepend;

			// Overloaded setters for 'x', 'y', and 'z' properties
			void (vector_array::* set_x_s)(const scalar_array&) = &vector_array::set_x;
			void (vector_array::* set_x_n)(numeric::array) = &vector_array::set_x;
			void (vector_array::* set_x_d)(double) = &vector_array::set_x;
			void (vector_array::* set_y_s)(const scalar_array&) = &vector_array::set_y;
			void (vector_array::* set_y_n)(numeric::array) = &vector_array::set_y;
			void (vector_array::* set_y_d)(double) = &vector_array::set_y;
			void (vector_array::* set_z_s)(const scalar_array&) = &vector_array::set_z;
			void (vector_array::* set_z_n)(numeric::array) = &vector_array::set_z;
			void (vector_array::* set_z_d)(double) = &vector_array::set_z;

			vector_array (vector_array::* truediv_double)( double) const = &vector_array::operator/;
			vector_array (vector_array::* truediv_scalar_array)(const scalar_array&) const = &vector_array::operator/;
			const vector_array& (vector_array::* itruediv_double)(double) = &vector_array::operator/=;
			const vector_array& (vector_array::* itruediv_scalar_array)(const scalar_array&) = &vector_array::operator/=;

			class_<vector_array> vector_array_wrapper( "vector_array", init< optional<int, vector> >( args("size", "fill")));
			vector_array_wrapper.def( init<const list&>())
			.def( init<numeric::array>())
			.def( self * double())
			.def( double() * self)
			.def( self * other<scalar_array>())
			.def( other<scalar_array>() * self)
			.def( self * other<vector_array>())
			.def( other<vector_array>() * self)
			.def( self * other<vector>())
			.def( self / double())
			.def( self / other<scalar_array>())
			.def( -self)
			.def( self *= double())
			.def( self *= other<scalar_array>())
			.def( self /= double())
			.def( self /= other<scalar_array>())
			.def( self + self)
			.def( self + other<vector>())
			.def( other<vector>() + self)
			.def( self - self)
			.def( self - other<vector>())
			.def( other<vector>() - self)
			.def( self += self)
			.def( self += other<vector>())
			.def( self -= self)
			.def( self -= other<vector>())
			.def( self >= self)
			.def( self >= other<scalar_array>())
			.def( self >= other<double>())
			.def( self <= self)
			.def( self <= other<scalar_array>())
			.def( self <= other<double>())
			.def( "__truediv__", truediv_double)
			.def( "__truediv__", truediv_scalar_array)
			.def( "__itruediv__", itruediv_double, return_value_policy<copy_const_reference>())
			.def( "__itruediv__", itruediv_scalar_array, return_value_policy<copy_const_reference>())
			.def( "proj", proj_vector)
			.def( "proj", proj_vector_array, "Returns a vector_array of vector projections of this array"
					" to a vector, or another vector_array of identical length.")
			.def( "cross", cross_vector)
			.def( "cross", cross_vector_array, "Returns a vector_array of cross products of this array"
					" and a vector, or another vector_array of identical length.")
			.def( "size", &vector_array::size, "Get the number of elements in the array.")
			.def( "rotate", &vector_array::rotate, vector_array_rotate( args("angle", "axis"), "Rotate a vector_array about an axis vector through an angle."))
			.def( "append", append_vector, "Add an element to the end of the array.")
			.def( "prepend", prepend_vector, "Add an element to the beginning of the array.")
			.def( "head_clip", &vector_array::head_clip, "Remove an element from the beginning of the arrray.")
			.def( "head_crop", &vector_array::head_crop, "Remove n elements from the beginning of the array.")
			.def( "tail_clip", &vector_array::tail_clip, "Remove an element from the end of the array.")
			.def( "tail_crop", &vector_array::tail_crop, "Remove n elements from the end of the array.")
			.def( "dot", dot_vector)
			.def( "dot", dot_vector_array, "Returns a scalar_array of dot products of this array"
					" and a vector, or another vector_array of identical length.")
			.def( "comp", comp_vector)
			.def( "comp", comp_vector_array, "Returns a scalar_array of scalar projections of this array"
					" to a vector, or another vector_array of identical length.")
			.def( "mag", &vector_array::mag, "Returns a scalar_array of the magnitudes of every element in this array.")
			.def( "mag2", &vector_array::mag2, "Equivalant to x.mag() * x.mag(), but faster.")
			.def( "norm", &vector_array::norm, "Returns a vector_array of the unit vectors of this array.")
			.def( "abs", &vector_array::fabs, "Returns a vector_array of absolute values of the vectors in the array.")
			// Fancy C++ -> python iterator access
			.def( "__iter__", iterator<vector_array>())
			.def( "__len__", &vector_array::size)
			.def( "__getitem__", &vector_array::py_getitem, return_internal_reference<>(), "Index this array by a single integer.\n"
					"Returns a reference to the indexed vector.")
			.def( "__setitem__", &vector_array::py_setitem)
			.def( "get_x", &vector_array::get_x)
			.def( "set_x", set_x_s)
			.def( "set_x", set_x_n)
			.def( "set_x", set_x_d)
			.def( "get_y", &vector_array::get_y)
			.def( "set_y", set_y_s)
			.def( "set_y", set_y_n)
			.def( "set_y", set_y_d)
			.def( "get_z", &vector_array::get_z)
			.def( "set_z", set_z_s)
			.def( "set_z", set_z_n)
			.def( "set_z", set_z_d)
			.def( "sum", &vector_array::sum, "Returns the sum of all elements in the array.")
			.def( "as_array", &vector_array::as_array, "Create a self.__len__() x 3 Numeric.array from this vector_array.")
			;

			vector_array_wrapper.add_property( "x", vector_array_wrapper.attr("get_x"),
					vector_array_wrapper.attr("set_x"));
			vector_array_wrapper.add_property( "y", vector_array_wrapper.attr("get_y"),
					vector_array_wrapper.attr("set_y"));
			vector_array_wrapper.add_property( "z", vector_array_wrapper.attr("get_z"),
					vector_array_wrapper.attr("set_z"));

			def( "sphere_intercollisions", &sphere_collisions, args( "pos", "radius"),
					"Evaluate collisions between spheres with centers == pos, and radii == radius.\n"
					"Returns a list of two-integer tuple indexes.\n"
					"The indexes corrispond to collision pairs indexed by pos.");
			def( "sphere_to_plane_collisions", &sphere_to_plane_collisions
					, args( "pos", "radius", "normal", "origin")
					, "Evaluate collisions between spheres with centers == pos, and radii == radius with\n"
					"a plane specified by a normal vector == normal, and a point on the plane == origin.\n"
					"Returns a list of integers corrisponding to colliding spheres' indexed into pos." );

		}

	}} // !namespace cvisual::python
