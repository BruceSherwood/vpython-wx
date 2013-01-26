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
							int old_x, int old_y,
							int new_x, int new_y,
							int shift_state_count, bool shift_state[] );

	// Get the current position of the mouse cursor relative to the window client area
	int get_x();
	int get_y();

	mouse_t& get_mouse();

 private:
	void update( bool new_buttons[], int old_px, int old_py, int new_px, int new_py, bool new_shift[] );

	mouse_t mouse;
	display_kernel& display;

	bool buttons[2];
	int px, py;
	bool left_down, left_dragging, left_semidrag;
	bool middle_down, middle_dragging, middle_semidrag;
	bool right_down, right_dragging, right_semidrag;
};

}  // namespace cvisual

#endif
