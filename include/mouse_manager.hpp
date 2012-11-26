#ifndef VPYTHON_MOUSE_MANAGER_HPP
#define VPYTHON_MOUSE_MANAGER_HPP
#pragma once

#include "mouseobject.hpp"

namespace cvisual {

// mouse_manager is reponsible for translating physical mouse movements into VPython
//   mouse events and camera actions.

class mouse_manager {
 public:
	mouse_manager( class display_kernel& display );

	// Called by the display driver to report mouse movement
	// Ideally this should be called in event handlers so that each successive change in mouse state
	// is captured in order, but a driver might just call this periodically with the current state.
	// If the mouse is locked, but has "moved" by (dx,dy), the driver should pass get_x()+dx, get_y()+dy
	void report_mouse_state(int physical_button_count, bool is_button_down[],
							int cursor_client_x, int cursor_client_y,
							int shift_state_count, bool shift_state[],
							bool driver_can_lock_mouse );

	// Get the current position of the mouse cursor relative to the window client area
	int get_x();
	int get_y();

	// On down button that initiates spin or zoom, save the location:
	int xlock;
	int ylock;

	mouse_t& get_mouse();

	void report_setcursor( int, int );

 private:
	void update( bool new_buttons[], int new_px, int new_py, bool new_shift[], bool can_lock );

	mouse_t mouse;
	display_kernel& display;

	bool buttons[2];
	int px, py;
	bool locked;
	int locked_px, locked_py;
	bool left_down, left_dragging, left_semidrag;
	bool middle_down, middle_dragging, middle_semidrag;
	bool right_down, right_dragging, right_semidrag;
};

}  // namespace cvisual

#endif
