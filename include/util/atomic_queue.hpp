#ifndef VPYTHON_UTIL_ATOMIC_QUEUE_HPP
#define VPYTHON_UTIL_ATOMIC_QUEUE_HPP

#include <queue>
#include <iostream>
#include <boost/python/import.hpp>

namespace cvisual {
	
// This class exists to separate out code that is not templated (and therefore
// doesn't need to be in a header file) from code that is.  This prevents
// picking up the whole Python runtine throughout all of cvisual, thus making
// compilation faster.
class atomic_queue_impl
{
 protected:
	volatile bool empty;
	
	atomic_queue_impl() : empty(true)
	{}

	void push_notify();
};

using boost::python::import;
using boost::python::object;

template <typename T>
class atomic_queue : private atomic_queue_impl
{
 private:
	std::queue<T> data;
	
	/**
	 * Precondition - at least one item is in data
	 */
	T pop_impl()
	{
		T ret = data.front(); // get the oldest element
		data.pop(); // remove the oldest element
		if (data.empty())
			empty = true;
		return ret;
	}

 public:
	atomic_queue() {}
	
	void push( const T& item)
	{
		data.push( item);
	}

	T peek() // assumes that data is not empty
	{
		return data.front();
	}

	T pop()
	{
		return pop_impl();
	}
	
	size_t size() const
	{
		return data.size();
	}
	
	void clear()
	{
		while (!data.empty())
			data.pop();
		empty = true;
	}
};

} // !namespace cvisual

#endif // !defined VPYTHON_UTIL_ATOMIC_QUEUE_HPP
