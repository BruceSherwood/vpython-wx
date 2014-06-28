#ifndef VPYTHON_DISPLAY_KERNEL_HPP
#define VPYTHON_DISPLAY_KERNEL_HPP

// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "renderable.hpp"
#include "util/vector.hpp"
#include "util/rgba.hpp"
#include "util/extent.hpp"
#include "util/gl_extensions.hpp"
#include "util/atomic_queue.hpp"
#include "mouse_manager.hpp"
#include "mouseobject.hpp"
#include <list>
#include <vector>
#include <set>
#include <string>

#include <boost/iterator/indirect_iterator.hpp>
#include <boost/tuple/tuple.hpp>

namespace cvisual {

using boost::indirect_iterator;

/** A class that manages all OpenGL aspects of a given scene.  This class
	requires platform-specific support from render_surface to manage an OpenGL
	rendering context and mouse and keyboard interaction.
*/
class display_kernel
{
 private: // Private data
 	shared_ptr<std::set<std::string> > extensions;
 	std::string renderer;
 	std::string version;
 	std::string vendor;
 	double last_time;
 	double render_time;
 	bool realized;

 	static shared_ptr<display_kernel> selected;

	shared_vector center; ///< The observed center of the display, in world space.
	shared_vector forward; ///< The direction of the camera, in world space.
	shared_vector up; ///< The vertical orientation of the scene, in world space.
	vector internal_forward; ///< Do not permit internal_forward to be +up or -up
	vector range; ///< Explicitly specified scene.range, or (0,0,0)
	vector camera; //< World coordinates of camera location
	double range_auto;	//< Automatically determined camera z from autoscale

	/** True initally and whenever the camera direction changes.  Set to false
	 * after every render cycle.
	 */
	bool forward_changed;

	extent_data world_extent; ///< The extent of the current world.

	double fov; ///< The field of view, in radians
	float stereodepth; //< How far in or out of the screen the scene seems to be
	bool autoscale; ///< True if Visual should scale the camera's position automatically.
	/** True if Visual should automatically reposition the center of the scene. */
	bool autocenter;
	/** True if the autoscaler should compute uniform axes. */
	bool uniform;
	/** A scaling factor determined by middle mouse button scrolling. */
	double user_scale;

	/** The global scaling factor. It is used to ensure that objects with
	 large dimensions are rendered properly. See the .cpp file for details.
	*/
	double gcf;
	/** Vector version of the global scaling factor used when scene.uniform=0.
	 Affects just curve, points, faces, label, frame, and conversion of mouse coordinates.
	*/
	vector gcfvec;

	/** True if the gcf has changed since the last render cycle.  Set to false
	 * after every rendering cycle.
	 */
	bool gcf_changed;

	rgb ambient; ///< The ambient light color.
	/** Called at the beginning of a render cycle to establish lighting. */
	void enable_lights(view& scene);
	/** Called at the end of a render cycle to complete lighting. */
	void disable_lights();

	rgb background; ///< The background color of the scene.
	rgb foreground; ///< The default color for objects to be rendered into the scene.

	// Whether or not the user is allowed to spin or zoom the display
	bool spin_allowed;
	bool zoom_allowed;

	/** Set up the OpenGL transforms from world space to view space. */
	void world_to_view_transform( view&, int whicheye = 0, bool forpick = false);
	/** Renders the scene for one eye.
		@param scene The dimensions of the scene, to be propogated to this
			display_kernel's children.
		@param eye Which eye is being rendered.  -1 for the left, 0 for the
			center, and 1 for the right.
		@param scene_geometry.anaglyph  True if using anaglyph stereo requiring color
			desaturation or grayscaling.
		@param scene_geometry.coloranaglyph  True if colors must be grayscaled, false if colors
			must be desaturated.
	*/
	bool draw( view&, int eye=0);

	/** Opaque objects to be rendered into world space. */
	std::list<shared_ptr<renderable> > layer_world;
	typedef indirect_iterator<std::list<shared_ptr<renderable> >::iterator> world_iterator;

	/** objects with a nonzero level of transparency that need to be depth sorted
		prior to rendering.
	*/
	std::vector<shared_ptr<renderable> > layer_world_transparent;
	typedef indirect_iterator<std::vector<shared_ptr<renderable> >::iterator> world_trans_iterator;

	// Computes the extent of the scene and takes action for autozoom and
	// autoscaling.
	void recalc_extent();

	// Compute the tangents of half the vertical and half the horizontal
	// true fields-of-view.
	void tan_hfov( double* x, double* y);

	void realize();
	void implicit_activate();

protected:
	// Mouse and keyboard objects
	mouse_manager mouse;

	// The bounding rectangle of the window on the screen (or equivalent super-window
	// coordinate system), including all decorations.
	// If the window is invisible, window_x and/or window_y may be -1, meaning
	// that the window will be positioned automatically by the window system.
	int window_x, window_y, window_width, window_height;

	// The rectangle on the screen into which we can actually draw.
	// At present, these are undefined until the display is realized, and
	// they are not used in constructing the display (they are outputs of
	// that process)
	// This includes both viewports in a side-by-side stereo mode, whereas
	//   view::view_width does not.
	int view_width, view_height;

	bool exit; ///< True when Visual should shutdown on window close.
	bool visible; ///< scene.visible
	bool explicitly_invisible;  ///< true iff scene.visible has ever been set to 0 by the program, or by the user closing a window
	bool fullscreen; ///< True when the display is in fullscreen mode.
	bool show_toolbar; ///< True when toolbar is displayed (pan, etc).
	std::string title;

public: // Public Data.
	gl_extensions glext;

	enum mouse_mode_t { ZOOM_ROTATE, ZOOM_ROLL, PAN, FIXED } mouse_mode;
	enum mouse_button { NONE, LEFT, RIGHT, MIDDLE };
	enum stereo_mode_t { NO_STEREO, PASSIVE_STEREO, ACTIVE_STEREO, CROSSEYED_STEREO,
		REDBLUE_STEREO, REDCYAN_STEREO, YELLOWBLUE_STEREO, GREENMAGENTA_STEREO
	} stereo_mode;

	/** Older machines should set this to some number between -6 and 0.  All of
		the tesselated models choose a lower level of detail based on this value
		when it is less than 0.
	*/
	int lod_adjust;

	/** Add a normal renderable object to the list of objects to be rendered into
	 *  world space.
	 */
	void add_renderable( shared_ptr<renderable>);

	/**  Remove a renderable object from this display, regardless of which layer
	 *   it resides in.  */
	void remove_renderable( shared_ptr<renderable>);

 public: // Public functions
	// Compute the location of the camera based on the current geometry.
	vector calc_camera();

	display_kernel();
	virtual ~display_kernel();

	/** Renders the scene once.  The enveloping widget is resposible for calling
		 this function appropriately.
 		@return If false, something catastrophic has happened and the
 		application should probably exit.
	*/
	bool render_scene();

	/** Inform this object that the window has been closed (is no longer physically
	    visible)
	*/
	void report_closed();

	/** Called by mouse_manager to report mouse movement that should affect the camera.
		Report that the mouse moved with one mouse button down.
 		@param dx horizontal change in mouse position in pixels.
 		@param dy vertical change in mouse position in pixels.
	*/
	void report_camera_motion( int dx, int dy, mouse_button button);

	/** Report that the position and/or size of the window or drawing area widget has changed.
		Some platforms might not know about position changes; they can pass (x,y,new_width,new_height)

 		win_* give the window rectangle (see this->window_*)
 		v_* give the view rectangle (see this->view_*)
 		*/
	void report_window_resize( int win_x, int win_y, int win_w, int win_h );
	void report_view_resize( int v_w, int v_h );

	/** Determine which object (if any) was picked by the cursor.
 	    @param x the x-position of the mouse cursor, in pixels.
		@param y the y-position of the mouse cursor, in pixels.
		@param d_pixels the allowable variation in pixels to successfully score
			a hit.
		@return  the nearest selected object, the position that it was hit, and
			the position of the mouse cursor on the near clipping plane.
           retval.get<0>() may be NULL if nothing was hit, in which case the
           positions are undefined.
	*/
	boost::tuple<shared_ptr<renderable>, vector, vector>
	pick( int x, int y, float d_pixels = 2.0);

	/** Recenters the scene.  Call this function exactly once to move the visual
	 * center of the scene to the true center of the scene.  This will work
	 * regardless of the value of this->autocenter.
	 */
	void recenter();

	/** Rescales the scene.  Call this function exactly once to scale the scene
	 * such that it fits within the entire window.  This will work
	 * regardless of the value of this->autoscale.
	 */
	void rescale();

	/** Release GL resources.  Call this as many times as you like during the
	 * shutdown.  However, neither pick() nor render_scene() may be called on
	 * any display_kernel after gl_free() has been invoked.
	 */
	void gl_free();

	void allow_spin(bool);
	bool spin_is_allowed(void) const;

	void allow_zoom(bool);
	bool zoom_is_allowed(void) const;


	// Python properties
	void set_up( const vector& n_up);
	shared_vector& get_up();

	void set_forward( const vector& n_forward);
	shared_vector& get_forward();

	void set_scale( const vector& n_scale);
	vector get_scale();

	void set_center( const vector& n_center);
	shared_vector& get_center();

	void set_fov( double);
	double get_fov();
	void set_lod(int);
	int get_lod();

	void set_uniform( bool);
	bool is_uniform();

	void set_background( const rgb&);
	rgb get_background();

	void set_foreground( const rgb&);
	rgb get_foreground();

	void set_autoscale( bool);
	bool get_autoscale();

	void set_autocenter( bool);
	bool get_autocenter();

	void set_range_d( double);
	void set_range( const vector&);
	vector get_range();

	void set_ambient_f( float);
	void set_ambient( const rgb&);
	rgb get_ambient();

	void set_stereodepth( float);
	float get_stereodepth();

	// The only mode that cannot be changed after initialization is active,
	// which will result in a gl_error exception when rendered.  The completing
	// display class will have to perform some filtering on this parameter.  This
	// properties setter will not change the mode if the new one is invalid.
	void set_stereomode( std::string mode);
	std::string get_stereomode();

	// A list of all objects rendered into this display_kernel.  Modifying it
	// does not propogate to the owning display_kernel.
	std::vector<shared_ptr<renderable> > get_objects() const;

	std::string info( void);

	void set_x( float x);
	float get_x();

	void set_y( float y);
	float get_y();

	void set_width( float w);
	float get_width();

	void set_height( float h);
	float get_height();

	void set_visible( bool v);
	bool get_visible();

	void set_title( std::string n_title);
	std::string get_title();

	bool is_fullscreen();
	void set_fullscreen( bool);

	bool get_exit();
	void set_exit(bool);

	bool is_showing_toolbar();
	void set_show_toolbar( bool);

	static bool enable_shaders;

	mouse_t* get_mouse();

	static void set_selected( shared_ptr<display_kernel> );
	static shared_ptr<display_kernel> get_selected();

	bool hasExtension( const std::string& ext );

	void pushkey(std::string k);

	typedef void (APIENTRYP EXTENSION_FUNCTION)();
	//virtual EXTENSION_FUNCTION getProcAddress( const char* );

	EXTENSION_FUNCTION getProcAddress( const char* );

	virtual void activate( bool active ) = 0;
};

} // !namespace cvisual

#endif // !defined VPYTHON_DISPLAY_KERNEL_HPP
