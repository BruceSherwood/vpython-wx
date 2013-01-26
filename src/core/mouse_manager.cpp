#include "mouse_manager.hpp"
#include "display_kernel.hpp"
#include <iostream>

namespace cvisual {

/* The implementation here is rather rudimentary - a step backward in functionality.
   But it provides a very simple interface to the display drivers, which should suffice
   to implement all the features we need.  So we can get the drivers right now and worry
   about the details of event handling later. */

mouse_manager::mouse_manager( class display_kernel& display )
 : display(display), px(0), py(0),
	 left_down(false), left_dragging(false), left_semidrag(false),
	 middle_down(false), middle_dragging(false), middle_semidrag(false),
	 right_down(false), right_dragging(false), right_semidrag(false)
{
	buttons[0] = buttons[1] = false;
}

// trivial properties
mouse_t& mouse_manager::get_mouse() { return mouse; }

static void fill( int out_size, bool out[], int in_size, bool in[], bool def = false ) {
	for(int i=0; i<out_size && i<in_size; i++)
		out[i] = in[i];
	for(int i=in_size; i<out_size; i++)
		out[i] = def;
}

int mouse_manager::get_x() {
	return px;
}

int mouse_manager::get_y() {
	return py;
}

void mouse_manager::report_mouse_state( int physical_button_count, bool is_button_down[],
										int old_x, int old_y,
										int new_x, int new_y,
										int shift_state_count, bool shift_state[] )
{
	// In is_button_down array, position 0=left, 1=right, 2=middle

	// A 2-button mouse with shift,ctrl,alt/option,command
	bool new_buttons[2]; fill(2, new_buttons, physical_button_count, is_button_down );
	bool new_shift[4]; fill(4, new_shift, shift_state_count, shift_state);

	// if we have a 3rd button, pressing it is like pressing both left and right buttons.
	// This is necessary to make platforms that "emulate" a 3rd button behave sanely with
	// a two-button mouse.
	if (physical_button_count>=3 && is_button_down[2])
		new_buttons[0] = new_buttons[1] = true;

	// If there's been more than one button change, impose an order, so that update()
	// only sees one change at a time
	// We choose the order so that the right button is down as much as possible, to
	// avoid spurious left button activity
	// Not relevant if there is a real (or emulated) button 2 (e.g. Mac option key)
	if (!new_buttons[2] && !buttons[2] && new_buttons[0] != buttons[0] && new_buttons[1] != buttons[1]) {
		int b = !new_buttons[1];
		new_buttons[b] = !new_buttons[b];
		update( new_buttons, old_x, old_y, new_x, new_y, new_shift );
		new_buttons[b] = !new_buttons[b];
	}
	
	update( new_buttons, old_x, old_y, new_x, new_y, new_shift );
}

void mouse_manager::update( bool new_buttons[], int old_px, int old_py, int new_px, int new_py, bool new_shift[] ) {
	// Shift states are just passed directly to mouseobject
	mouse.set_shift( new_shift[0] );
	mouse.set_ctrl( new_shift[1] );
	mouse.set_alt( new_shift[2] );
	mouse.set_command( new_shift[3] );

	/*
	if (can_lock_mouse) { // we are at start of spin or zoom
		px = new_px;
		py = new_py;
	}

	bool was_locked = locked;
	locked = (can_lock_mouse && display.zoom_is_allowed() && new_buttons[0] && new_buttons[1]) ||
	         (can_lock_mouse && display.spin_is_allowed() && new_buttons[1] && !new_buttons[0]);
	if (locked && !was_locked) { locked_px = new_px; locked_py = new_py; }
	*/

	if (new_buttons[1]) // handle spin or zoom if allowed
		display.report_camera_motion( (new_px - old_px), (new_py - old_py),
									  new_buttons[0] ? display_kernel::MIDDLE : display_kernel::RIGHT );

	// left_semidrag means that we've moved the mouse and so can't get a left click, but we aren't
	// necessarily actually dragging, because the movement might have occurred with the right button down.
	if (left_down && !left_dragging && (new_px != old_px || new_py != old_py))
		left_semidrag = true;
	if (!left_down) left_semidrag = false;

	if (!display.spin_is_allowed()) {
		if (right_down && !right_dragging && (new_px != old_px || new_py != py))
			right_semidrag = true;
		if (!right_down) right_semidrag = false;
	}

	if (!display.zoom_is_allowed()) {
		if (middle_down && !middle_dragging && (new_px != old_px || new_py != py))
			middle_semidrag = true;
		if (!middle_down) middle_semidrag = false;
	}

	// In reporting with press_event etc., 1=left, 2=right, 3=middle

	if (!new_buttons[1]) { //< Ignore changes in the left button state while the right button is down!
		bool b = new_buttons[0];

		if (b != left_down) {
			if (b) {
				if ( !buttons[0] ) //< Releasing the other button of a chord doesn't "press" the left
					mouse.push_event( press_event(1, mouse) );
				else
					b = false;
			} else if ( left_dragging ) {
				mouse.push_event( drop_event(1, mouse) );
				left_dragging = false;
			} else if ( left_semidrag ) {
				mouse.push_event( release_event(1, mouse) );
			} else if (left_down) {
				mouse.push_event( click_event(1, mouse) );
			}
		}

		if ( b && left_down && (new_px != old_px || new_py != py) && !left_dragging ) {
			mouse.push_event( drag_event(1, mouse) );
			left_dragging = true;
		}

		left_down = b;
	}
	if (!display.spin_is_allowed() && !new_buttons[0]) { //< Ignore changes in the left button state while the right button is down!
		bool b = new_buttons[1];

		if (b != right_down) {
			if (b) {
				if ( !buttons[1] ) //< Releasing the other button of a chord doesn't "press" the right
					mouse.push_event( press_event(2, mouse) );
				else
					b = false;
			} else if ( right_dragging ) {
				mouse.push_event( drop_event(2, mouse) );
				right_dragging = false;
			} else if ( right_semidrag ) {
				mouse.push_event( release_event(2, mouse) );
			} else if (right_down) {
				mouse.push_event( click_event(2, mouse) );
			}
		}

		if ( b && right_down && (new_px != old_px || new_py != py) && !right_dragging ) {
			mouse.push_event( drag_event(2, mouse) );
			right_dragging = true;
		}

		right_down = b;
	}
	if (!display.zoom_is_allowed()) {
		bool b = (new_buttons[0] && new_buttons[1]);

		if (b != middle_down) {
			if (b) {
				if ( !(buttons[0] && buttons[1]) )
					mouse.push_event( press_event(3, mouse) );
				else
					b = false;
			} else if ( middle_dragging ) {
				mouse.push_event( drop_event(3, mouse) );
				middle_dragging = false;
			} else if ( middle_semidrag ) {
				mouse.push_event( release_event(3, mouse) );
			} else if (middle_down) {
				mouse.push_event( click_event(3, mouse) );
			}
		}

		if ( b && middle_down && (new_px != old_px || new_py != py) && !middle_dragging ) {
			mouse.push_event( drag_event(3, mouse) );
			middle_dragging = true;
		}

		middle_down = b;
	}

	px = new_px;
	py = new_py;

	for(int b=0; b<2; b++) buttons[b] = new_buttons[b];
}

} // namespace cvisual
