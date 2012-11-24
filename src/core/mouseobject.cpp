// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "mouseobject.hpp"
#include "display_kernel.hpp"
#include <stdexcept>

#include "util/rate.hpp"

namespace cvisual {

static void init_event( int which, shared_ptr<event> ret, const mouse_t& mouse)
{
	ret->position = mouse.position;
	ret->pick = mouse.pick;
	ret->pickpos = mouse.pickpos;
	ret->cam = mouse.cam;
	ret->set_shift( mouse.is_shift());
	ret->set_ctrl( mouse.is_ctrl());
	ret->set_alt( mouse.is_alt());
	ret->set_command( mouse.is_command());
	switch (which) {
		case 1:
			ret->set_leftdown( true);
			break;
		case 2:
			ret->set_rightdown( true);
			break;
		case 3:
			ret->set_middledown( true);
			break;
		default:
			bool button_is_known = false;
			assert( button_is_known == true);
	}
}

shared_ptr<event>
press_event( int which, const mouse_t& mouse)
{
	shared_ptr<event> ret( new event());;
	ret->set_press( true);
	init_event( which, ret, mouse);
	return ret;
}

shared_ptr<event>
drop_event( int which, const mouse_t& mouse)
{
	shared_ptr<event> ret( new event());;
	ret->set_release( true);
	ret->set_drop( true);
	init_event( which, ret, mouse);
	return ret;
}

shared_ptr<event>
release_event( int which, const mouse_t& mouse)
{
	shared_ptr<event> ret( new event());;
	ret->set_release( true);
	init_event( which, ret, mouse);
	return ret;
}

shared_ptr<event>
click_event( int which, const mouse_t& mouse)
{
	shared_ptr<event> ret( new event());;
	ret->set_release( true);
	ret->set_click( true);
	init_event( which, ret, mouse);
	return ret;
}

shared_ptr<event>
drag_event( int which, const mouse_t& mouse)
{
	shared_ptr<event> ret( new event());;
	ret->set_drag( true);
	init_event( which, ret, mouse);
	return ret;
}

mousebase::~mousebase()
{
}

/* Translate a button click code into a text string.
 */
std::string*
mousebase::get_buttons() const
{
	if (buttons.test( left))
		return new std::string( "left");
	else if (buttons.test( right))
		return new std::string( "right");
	else if (buttons.test( middle))
		return new std::string( "middle");
	else
		return 0;
}

/* Project the cursor's current location onto the plane specified by the normal
 * vector 'normal' and a perpendicular distance 'dist' from the origin.
 */
vector
mousebase::project1( vector normal, double dist)
{
	double ndc = normal.dot(cam) - dist;
	double ndr = normal.dot(get_ray());
	double t = -ndc / ndr;
	vector v = cam + get_ray()*t;
	return v;
}

/* Project the cursor's current position onto the plane specified by the normal vector
 * 'normal' rooted at the position vector 'point'.
 */
vector
mousebase::project2( vector normal, vector point)
{
	double dist = normal.dot(point);
	double ndc = normal.dot(cam) - dist;
	double ndr = normal.dot(get_ray());
	double t = -ndc / ndr;
	vector v = cam + get_ray()*t;
	return v;
}

shared_ptr<renderable>
mousebase::get_pick()
{
	return pick;
}

/************** event implementation **************/
#if 0
shared_ptr<event>
event::create_press(
		shared_ptr<renderable> pick,
		vector pickpos,
		modifiers buttonstate,
		display_kernel::mouse_button buttons)
{
	shared_ptr<event> ret( new event());
	ret->pickpos = pickpos;
	ret->pick = pick;
	if (buttonstate & ctrl)
		ret->set_ctrl( true);
	if (buttonstate & shift)
		ret->set_shift( true);
	if (buttonstate & alt)
		ret->set_alt();
	ret->set_press( true);
	ret->buttons = buttons;

	return ret;
}
#endif

/************** mouseObject implementation **************/

mouse_t::~mouse_t()
{
}

void
mouse_t::clear_events( int i)
{
	if (i != 0) {
		throw std::invalid_argument( "mouse.events can only be set to zero");
	}
	events.clear();
	return;
}

int
mouse_t::num_events() const
{
	return events.size();
}

int
mouse_t::num_clicks() const
{
	return click_count;
}


shared_ptr<event>
mouse_t::peek_event() // this is scene.mouse.peekevent()
{
	shared_ptr<event> ret = events.peek();
	return ret;
}

void
mouse_t::push_event( shared_ptr<event> e)
{
	if (e->is_click())
		click_count++;
	//std::cout << "mouseobject.cpp push_event click_count = " << click_count << std::endl;
	events.push( e);
}

// http://bfroehle.com/2011/07/boost-python-and-boost-function/

boost::function<wait_t> wait;

void set_wait(object obj) {
	wait = obj;
}

void call_wait() {
	wait(0.03);
}

shared_ptr<event>
mouse_t::pop_event() // this is scene.mouse.getevent()
{
	// In VPython 5.x, the while loop was interrupted by the render thread
	// In VPython 6.x, the while loop needs a wait statement to get events
	shared_ptr<event> ret;
	while (true) {
		if (events.size() > 0) {
			ret = events.pop();
			if (ret->is_click()) {
				click_count--;
			}
			return ret;
		}
		call_wait();
	}
}

shared_ptr<event>
mouse_t::pop_click() // this is scene.mouse.getclick()
{
	// In VPython 5.x, the while loop was interrupted by the render thread
	// In VPython 6.x, the while loop needs a wait statement to get events
	shared_ptr<event> ret;
	while (true) {
		while (events.size() > 0) {
			ret = events.pop();
			if (ret->is_click()) {
				click_count--;
				return ret;
			}
		}
		call_wait();
	}
}

} // !namespace visual
