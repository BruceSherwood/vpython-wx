// Copyright (c) 2000, 2001, 2002, 2003 by David Scherer and others.
// Copyright (c) 2003, 2004 by Jonathan Brandmeyer and others.
// See the file license.txt for complete license terms.
// See the file authors.txt for a complete list of contributors.

#include "display_kernel.hpp"
#include "util/errors.hpp"
#include "util/tmatrix.hpp"
#include "util/gl_enable.hpp"
#include "material.hpp"
#include "frame.hpp"
#include "wrap_gl.hpp"

#include <cassert>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <iostream>
#include <boost/scoped_array.hpp>

#include <boost/lexical_cast.hpp>

namespace cvisual {

shared_ptr<display_kernel> display_kernel::selected;

bool display_kernel::enable_shaders = true;

// TODO: The actual list and count of displays is now in Python code
static int displays_visible = 0;
void set_display_visible( display_kernel*, bool visible ) {
	if (visible) displays_visible++;
	else displays_visible--;
}

static const display_kernel::EXTENSION_FUNCTION notImplemented = (display_kernel::EXTENSION_FUNCTION)-1;

void
display_kernel::enable_lights(view& scene)
{
	scene.light_count[0] = 0;
	scene.light_pos.clear();
	scene.light_color.clear();
	std::list<shared_ptr<renderable> >::iterator i = layer_world.begin();
	std::list<shared_ptr<renderable> >::iterator i_end = layer_world.end();
	for(; i != i_end; ++i)
		(*i)->render_lights( scene );
	std::vector<shared_ptr<renderable> >::iterator j = layer_world_transparent.begin();
	std::vector<shared_ptr<renderable> >::iterator j_end = layer_world_transparent.end();
	for(; j != j_end; ++j)
		(*j)->render_lights( scene );

	tmatrix world_camera; world_camera.gl_modelview_get();
	vertex p;

	// Clear modelview matrix since we are multiplying the light positions ourselves
	gl_matrix_stackguard guard;
	glLoadIdentity();

	for(int i=0; i<scene.light_count[0] && i<8; i++) {
		int li = i*4;

		// Transform the light into eye space
		for(int d=0; d<4; d++) p[d] = scene.light_pos[li+d];
		p = world_camera * p;
		for(int d=0; d<4; d++) scene.light_pos[li+d] = p[d];

		// Enable the light for fixed function lighting.  This is unnecessary if everything in the scene
		// uses materials and the card supports our shaders, but for now...
		int id = GL_LIGHT0 + i;
		glLightfv( id, GL_DIFFUSE, &scene.light_color[li]);
		glLightfv( id, GL_SPECULAR, &scene.light_color[li]);
		glLightfv( id, GL_POSITION, &scene.light_pos[li]);
		glEnable(id);
	}
	for(int i=scene.light_count[0]; i<8; i++)
		glDisable( GL_LIGHT0 + i );

	glEnable( GL_LIGHTING);
	glLightModelfv( GL_LIGHT_MODEL_AMBIENT, &ambient.red);

	check_gl_error();
}

void
display_kernel::disable_lights()
{
	glDisable( GL_LIGHTING);
}

// Compute the horizontal and vertial tangents of half the field-of-view.
void
display_kernel::tan_hfov( double* x, double* y)
{
	// tangent of half the field of view.
	double tan_hfov = std::tan( fov*0.5);
	double aspect_ratio = (double)view_height / view_width;
	if (stereo_mode == PASSIVE_STEREO || stereo_mode == CROSSEYED_STEREO)
		aspect_ratio *= 2.0;
	if (aspect_ratio > 1.0) {
		// Tall window
		*x = tan_hfov / aspect_ratio;
		*y = tan_hfov;
	}
	else {
		// Wide window
		*x = tan_hfov;
		*y = tan_hfov * aspect_ratio;
	}
}

vector
display_kernel::calc_camera()
{
	return camera;
	/* old scheme not necessary?
	double tan_hfov_x = 0.0;
	double tan_hfov_y = 0.0;
	tan_hfov( &tan_hfov_x, &tan_hfov_y);
	double cot_hfov = 1 / std::min(tan_hfov_x, tan_hfov_y);
	return (-forward.norm() * cot_hfov*user_scale).scale(range) + center;
	*/
}

display_kernel::display_kernel()
	:
	exit(true),
	visible(false),
	explicitly_invisible(false),
	fullscreen(false),
	title( "VPython" ),
	window_x(0), window_y(0), window_width(430), window_height(450),
	view_width(-1), view_height(-1),
	center(0, 0, 0),
	forward(0, 0, -1),
	internal_forward(0, 0, -1),
	up(0, 1, 0),
	forward_changed(true),
	fov( 60 * M_PI / 180.0),
	autoscale(true),
	autocenter(false),
	uniform(true),
	camera(0,0,0),
	user_scale(1.0),
	gcf(1.0),
	gcfvec(vector(1.0,1.0,1.0)),
	gcf_changed(false),
	ambient( 0.2f, 0.2f, 0.2f),
	show_toolbar( false),
	last_time(0),
	background(0, 0, 0), //< Transparent black.
	spin_allowed(true),
	zoom_allowed(true),
	mouse_mode( ZOOM_ROTATE),
	stereo_mode( NO_STEREO),
	stereodepth( 0.0f),
	lod_adjust(0),
	realized(false),
	mouse( *this ),
	range_auto(0.0),
	range(0,0,0),
	world_extent(0.0)
{
}

display_kernel::~display_kernel()
{
	if (visible)
		set_display_visible( this, false );
}

void
display_kernel::report_closed() {
	if (visible)
		set_display_visible( this, false );

	realized = false;
	visible = false;
	explicitly_invisible = true;
}

void
display_kernel::report_camera_motion( int dx, int dy, mouse_button button )
{
	// This stuff handles automatic movement of the camera in response to user
	// input. See also view_to_world_transform for how the affected variables
	// are used to actually position the camera.

	// Scaling conventions:
	// the full width of the widget rotates the scene horizontally by 120 degrees.
	// the full height of the widget rotates the scene vertically by 120 degrees.
	// the full height of the widget zooms the scene by a factor of 10

	// Panning conventions:
	// The full height or width of the widget pans the scene by the eye distance.

	// Locking:
	// center and forward are already synchronized. The only variable that
	// remains to be synchronized is user_scale.

	// The vertical and horizontal fractions of the window's height that the
	// mouse has traveled for this event.
	// TODO: Implement ZOOM_ROLL modes.
	float vfrac = (float)dy / view_height;
	float hfrac = dx
		/ ((stereo_mode == PASSIVE_STEREO || stereo_mode == CROSSEYED_STEREO) ?
		     (view_width*0.5f) : view_width);

	// The amount by which the scene should be shifted in response to panning
	// motion.
	// TODO: Keep this synchronized with the eye_dist calc in
	// world_view_transform
	double tan_hfov_x = 0.0;
	double tan_hfov_y = 0.0;
	tan_hfov( &tan_hfov_x, &tan_hfov_y);
	double pan_rate = (center - calc_camera()).mag()
		* std::min( tan_hfov_x, tan_hfov_y);

	switch (button) {
		case NONE: case LEFT:
			break;
		case MIDDLE:
			switch (mouse_mode) {
				case FIXED:
					// Locked.
					break;
				case PAN:
					// Pan front/back.
					if (spin_allowed)
						center += pan_rate * vfrac * internal_forward.norm();
					break;
				case ZOOM_ROLL: case ZOOM_ROTATE:
					// Zoom in/out.
					if (zoom_allowed)
						user_scale *= std::pow( 10.0f, vfrac);
					break;
			}
			break;
		case RIGHT:
			switch (mouse_mode) {
				case FIXED: case ZOOM_ROLL:
					break;
				case PAN: {
					// Pan up/down and left/right.
					// A vector pointing along the camera's horizontal axis.
					vector horiz_dir = internal_forward.cross(up).norm();
					// A vector pointing along the camera's vertical axis.
					vector vert_dir = horiz_dir.cross(internal_forward).norm();
					if (spin_allowed) {
						center += -horiz_dir * pan_rate * hfrac;
						center += vert_dir * pan_rate * vfrac;
					}
					break;
				}
				case ZOOM_ROTATE: {
					if (spin_allowed) {
						// Rotate
						// First perform the rotation about the up vector.
						tmatrix R = rotation( -hfrac * 2.0, up.norm());
						internal_forward = R * internal_forward;

						// Then perform rotation about an axis orthogonal to up and forward.
						double vertical_angle = vfrac * 2.0;
						double max_vertical_angle = up.diff_angle(-internal_forward.norm());

						// Over the top (or under the bottom) rotation
						if (!(vertical_angle >= max_vertical_angle ||
							vertical_angle <= max_vertical_angle - M_PI)) {
							// Over the top (or under the bottom) rotation
							R = rotation( -vertical_angle, internal_forward.cross(up).norm());
							forward = internal_forward = R*internal_forward;
							forward_changed = true;
						}
					}
					break;
				}
			}
			break;
	}
}

void
display_kernel::report_window_resize( int win_x, int win_y, int win_w, int win_h )
{
	window_x = win_x; window_y = win_y; window_width = win_w; window_height = win_h;
}

void
display_kernel::report_view_resize(	int v_w, int v_h )
{
	view_width = std::max(v_w,1); view_height = std::max(v_h,1);
}

void
display_kernel::realize()
{
	clear_gl_error();
	if (!extensions) {
		using namespace std;
		extensions.reset( new set<string>());
		istringstream strm( string( (const char*)(glGetString( GL_EXTENSIONS))));
		copy( istream_iterator<string>(strm), istream_iterator<string>(),
			inserter( *extensions, extensions->begin()));

		vendor = std::string((const char*)glGetString(GL_VENDOR));
		version = std::string((const char*)glGetString(GL_VERSION));
		renderer = std::string((const char*)glGetString(GL_RENDERER));

		// The test is a hack so that subclasses not bothering to implement getProcAddress just
		//   don't get any extensions.
		if (getProcAddress("display_kernel::getProcAddress") != notImplemented)
			glext.init( *this );
	}

	// Those features of OpenGL that are always used are set up here.
	// Depth buffer properties
	glClearDepth( 1.0);
	glEnable( GL_DEPTH_TEST);
	glDepthFunc( GL_LEQUAL);

	// Lighting model properties
	glShadeModel( GL_SMOOTH);
	// TODO: Figure out what the concrete costs/benefits of these commands are.
	// glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	glHint( GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint( GL_POINT_SMOOTH_HINT, GL_NICEST);
	glEnable( GL_NORMALIZE);
	glColorMaterial( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable( GL_COLOR_MATERIAL);
	glEnable( GL_BLEND );
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Ensures that fully transparent pixels don't write into the depth buffer,
	// ever.
	glEnable( GL_ALPHA_TEST);
	glAlphaFunc( GL_GREATER, 0.0);

	// FSAA.  Doesn't seem to have much of an effect on my TNT2 card.  Grrr.
	if (hasExtension( "GL_ARB_multisample" ) ) {
		glEnable( GL_MULTISAMPLE_ARB);
		GLint n_samples, n_buffers;
		glGetIntegerv( GL_SAMPLES_ARB, &n_samples);
		glGetIntegerv( GL_SAMPLE_BUFFERS_ARB, &n_buffers);
		//VPYTHON_NOTE( "Using GL_ARB_multisample extension: samples:"
		//	+ boost::lexical_cast<std::string>(n_samples)
		//	+ " buffers: " + boost::lexical_cast<std::string>(n_buffers));
	}

	check_gl_error();
}

// Set up matrices for transforms from world coordinates to view coordinates
// Precondition: the OpenGL Modelview and Projection matrix stacks should be
// at the bottom.
// Postcondition: active matrix stack is GL_MODELVIEW, matrix stacks are at
// the bottom.  Viewing transformations have been applied.  geometry.camera
// is initialized.
// whicheye: -1 for left, 0 for center, 1 for right.
void
display_kernel::world_to_view_transform(
	view& geometry, int whicheye, bool forpick)
{
	// See http://www.stereographics.com/support/developers/pcsdk.htm for a
	// discussion regarding the design basis for the frustum offset code.

	// gcf scales the region encompassed by scene.range_* into a ROUGHLY 2x2x2 cube.
	// Note that this is NOT necessarily the entire world, since scene.range
	//   can be changed.
	// This coordinate system is used for most of the calculations below.

	vector scene_center = center.scale(gcfvec);
	vector scene_up = up.norm();
    vector scene_forward = internal_forward.norm();

	// the horizontal and vertical tangents of half the field of view.
	double tan_hfov_x;
	double tan_hfov_y;
	tan_hfov( &tan_hfov_x, &tan_hfov_y);

	// The cotangent of half of the wider field of view.
	double cot_hfov;
	if (!uniform) // We force width to be 2.0 (range.x 1.0)
		cot_hfov = 1.0 / tan_hfov_x;
	else
		cot_hfov = 1.0 / std::max(tan_hfov_x, tan_hfov_y);

	// The camera position is chosen by the tightest of the enabled range_* modes.
	double cam_to_center_without_zoom = 1e150;
	/*if (range_sphere_radius)
		cam_to_center_without_zoom = std::min(cam_to_center_without_zoom,
			range_sphere_radius / sin( fov * 0.5 ) );
	if (range_box_size.nonzero()) {
		if (range_unrotated) {
			cam_to_center_without_zoom = std::min(cam_to_center_without_zoom,
				std::max(range_box_size.x, range_box_size.y) * 0.5 * cot_hfov + range_box_size.z * 0.5);
		} else
			cam_to_center_without_zoom = std::min(cam_to_center_without_zoom,
				range_box_size.mag() * 0.5 / sin( fov * 0.5 ) );
	}*/
	if (range_auto)
		cam_to_center_without_zoom = std::min(cam_to_center_without_zoom,
			range_auto);
	if (range.nonzero())
		cam_to_center_without_zoom = std::min(cam_to_center_without_zoom,
			range.x * cot_hfov / 1.02);
	if (cam_to_center_without_zoom >= 1e150)
		cam_to_center_without_zoom = 10.0 / sin( fov * 0.5 );
	cam_to_center_without_zoom *= gcf * 1.02;

	// Position camera so that a sphere containing the box range will fit on the screen
	//   OR a 2*user_scale cube will fit.  The former is tighter for "non cubical" ranges
	//   and the latter is tighter for cubical ones.
	/*double radius = range.mag() * gcf * user_scale;
	double cam_to_center_without_zoom = 1.02 * std::min( radius / sin( fov * 0.5 ),
		                                                 cot_hfov + 1.0 );*/

	vector scene_camera = scene_center - cam_to_center_without_zoom*user_scale*scene_forward;

	double nearest, farthest;
	world_extent.get_near_and_far(internal_forward, nearest, farthest); // nearest and farthest points relative to scene.center when projected onto forward
	nearest = nearest*gcf;
	farthest = farthest*gcf;

	double cam_to_center = (scene_center - scene_camera).mag();
	// Z buffer resolution is highly sensitive to nearclip - a "small" camera will have terrible z buffer
	//   precision for distant objects.  PLEASE don't fiddle with this unless you know what kind of
	//   test cases you need to see the results, including at nonstandard fields of view and 24 bit
	//   z buffers!
	// The equation for nearclip below is designed to give similar z buffer resolution at all fields of
	//   view.  It's a little weird, but seems to give acceptable results in all the cases I've been able
	//   to test.
	// The other big design question here is the effect of "zoom" (user_scale) on the near clipping plane.
	//   Most users will have the mental model that this moves the camera closer to the scene, rather than
	//   scaling the scene up.  There is actually a difference since the camera has a finite "size".
	//   Unfortunately, following this model leads to a problem with zooming in a lot!  The problem is
	//   especially pronounced at tiny fields of view, which typically have an enormous camera very far away;
	//   when you try to zoom in the big camera "crashes" into the tiny scene!  So instead we use the
	//   slightly odd model of scaling the scene, or equivalently making the camera smaller as you zoom in.
	double fwz = cam_to_center_without_zoom + 1.0;
	double nearclip = fwz * fwz / (100 + fwz) * user_scale;
	// TODO: nearclip = std::max( nearclip, (cam_to_center + nearest) * 0.95 );  //< ?? boost z buffer resolution if there's nothing close to camera?
	double farclip = (farthest + cam_to_center) * 1.05;  //< actual maximum z in scene plus a little
	farclip = std::max( farclip, nearclip * 1.001 ); //< just in case everything is behind the camera!

	// Here is the stereodepth and eye offset machinery from Visual 3, where the docs claimed that
	// stereodepth=0 was the default (zero-parallax plane at screen surface;
	// stereodepth=1 moves the center of the scene to the screen surface;
	// stereodepth=2 moves the back of the scene to the screen surface:
	/*
	double farclip = cotfov + ext;
	double nearclip = 0.0;
	if ((cam - display->c_center).mag() < display->c_extent.mag()) {
		// Then the camera is within the scene.  Pick a value that looks OK.
		nearclip = 0.015;
	}
	else {
		nearclip = cotfov - ext*1.5;
		if (nearclip < 0.01*farclip)
			nearclip = 0.01*farclip;
	}
	double R = nearclip*hfov;
	double T = nearclip*vfov;

	double fl = 0.5*ext + ext*stereodepth + nearclip;  //focal length
	double eyeOffset = eyesign*fl/60.0;  // eye separation 1/30 of focallength
	double eyeOffset1 = eyeOffset * (nearclip/fl);
	frustum(proj, iproj, -R-eyeOffset1, R-eyeOffset1, -T, T, nearclip, farclip);
	*/

	// A multiple of the number of cam_to_center's away from the camera to place
	// the zero-parallax plane.
	// The distance from the camera to the zero-parallax plane.
	double focallength = cam_to_center+0.5*stereodepth;
	// Translate camera left/right 2% of the viewable width of the scene at
	// the distance of its center.
	//double camera_stereo_offset = tan_hfov_x * cam_to_center * 0.02;
	double camera_stereo_offset = tan_hfov_x * focallength * 0.02;
	vector camera_stereo_delta = camera_stereo_offset
		* up.cross( scene_camera).norm() * whicheye;
	scene_camera += camera_stereo_delta;
	scene_center += camera_stereo_delta;
	// The amount to translate the frustum to the left and right.
	double frustum_stereo_offset = camera_stereo_offset * nearclip
		/ focallength * whicheye;

	// Finally, the OpenGL transforms based on the geometry just calculated.
	clear_gl_error();
	// Position the camera.
	glMatrixMode( GL_MODELVIEW);
	glLoadIdentity();

	#if 0	// Enable this to peek at the actual scene geometry.
	int max_proj_stack_depth = -1;
	int max_mv_stack_depth = -1;
	int proj_stack_depth = -1;
	int mv_stack_depth = -1;
	glGetIntegerv( GL_MAX_PROJECTION_STACK_DEPTH, &max_proj_stack_depth);
	glGetIntegerv( GL_MAX_MODELVIEW_STACK_DEPTH, &max_mv_stack_depth);
	glGetIntegerv( GL_PROJECTION_STACK_DEPTH, &proj_stack_depth);
	glGetIntegerv( GL_MODELVIEW_STACK_DEPTH, &mv_stack_depth);
	std::cerr << "scene_geometry: camera:" << scene_camera
        << " true camera:" << camera << std::endl
		<< " center:" << scene_center << " true center:" << center << std::endl
		<< " forward:" << scene_forward << " true forward:" << forward << std::endl
		<< " up:" << scene_up << " range:" << range << " gcf:" << gcf  << std::endl
		<< " nearclip:" << nearclip << " nearest:" << nearest << std::endl
		<< " farclip:" << farclip << " farthest:" << farthest << std::endl
		<< " user_scale:" << user_scale << std::endl
        << " cot_hfov:" << cot_hfov << " tan_hfov_x:" << tan_hfov_x << std::endl
        << " tan_hfov_y: " << tan_hfov_y << std::endl
        << " window_width:" << window_width << " window_height:" << window_height << std::endl
        << " max_proj_depth:" << max_proj_stack_depth << " current_proj_depth:" << proj_stack_depth << std::endl
        << " max_mv_depth:" << max_mv_stack_depth << " current_mv_depth:" << mv_stack_depth << std::endl;
	world_extent.dump_extent();
	std::cerr << std::endl;
	#endif

	gluLookAt(
		scene_camera.x, scene_camera.y, scene_camera.z,
		scene_center.x, scene_center.y, scene_center.z,
		scene_up.x, scene_up.y, scene_up.z);

	tmatrix world_camera; world_camera.gl_modelview_get();
	inverse( geometry.camera_world, world_camera );

	//vector scene_range = range * gcf;
	//glScaled( 1.0/scene_range.x, 1.0/scene_range.y, 1.0/scene_range.z);

	// Establish a parallel-axis asymmetric stereo projection frustum.
	glMatrixMode( GL_PROJECTION);
	if (!forpick)
		glLoadIdentity();
	if (whicheye == 1) {
		frustum_stereo_offset = -frustum_stereo_offset;
	}
	else if (whicheye == 0) {
		frustum_stereo_offset = 0;
	}

	if (nearclip<=0 || farclip<=nearclip || tan_hfov_x<=0 || tan_hfov_y<=0) {
		std::ostringstream msg;
		msg << "VPython degenerate projection: " << nearclip << " " << farclip << " " << tan_hfov_x << " " << tan_hfov_y;
		VPYTHON_CRITICAL_ERROR( msg.str());
		std::exit(1);
	}

	glFrustum(
		-nearclip * tan_hfov_x + frustum_stereo_offset,
		nearclip * tan_hfov_x + frustum_stereo_offset,
		-nearclip * tan_hfov_y,
		nearclip * tan_hfov_y,
		nearclip,
		farclip );

	glMatrixMode( GL_MODELVIEW);
	check_gl_error();

	// The true camera position, in world space.
	camera = scene_camera/gcf;

	// Finish initializing the view object.
	geometry.camera = camera;
	geometry.tan_hfov_x = tan_hfov_x;
	geometry.tan_hfov_y = tan_hfov_y;
	// The true viewing vertical direction is not the same as what is needed for
	// gluLookAt().
	geometry.up = internal_forward.cross_b_cross_c(up, internal_forward).norm();
}

// Calculate a new extent for the universe, adjust gcf, center, and world_scale
// as required.
void
display_kernel::recalc_extent(void)
{
	double tan_hfov_x;
	double tan_hfov_y;
	tan_hfov( &tan_hfov_x, &tan_hfov_y );
	double tan_hfov = std::max(tan_hfov_x, tan_hfov_y);

	while (1) {  //< Might have to do this twice for autocenter
		world_extent = extent_data( tan_hfov );

		tmatrix l_cw;
		l_cw.translate( -center );
		extent ext( world_extent, l_cw );

		world_iterator i( layer_world.begin());
		world_iterator end( layer_world.end());
		while (i != end) {
			i->grow_extent( ext);
			++i;
		}
		world_trans_iterator j( layer_world_transparent.begin());
		world_trans_iterator j_end( layer_world_transparent.end());
		while (j != j_end) {
			j->grow_extent( ext);
			++j;
		}
		if (autocenter) {
			vector c = world_extent.get_center() + center;
			if ( (center-c).mag2() > (center.mag2() + c.mag2()) * 1e-6 ) {
				// Change center and recalculate extent (since camera_z depends on center)
				center = c;
				continue;
			}
		}
		break;
	}
	if (autoscale && uniform) {
		double r = world_extent.get_camera_z();
		if (r > range_auto) range_auto = r;
		else if ( 3.0*r < range_auto ) range_auto = 3.0*r;
	}

	// Rough scale calculation for gcf.  Doesn't need to be exact.
	// TODO: If extent and range are very different in scale, we are using extent to drive
	//   gcf.  Both options have pros and cons.
	double mr = world_extent.get_range(vector(0,0,0)).mag();
	double scale = mr ? 1.0 / mr : 1.0;

	if (!uniform && range.nonzero()) {
		gcf_changed = true;
		gcf = 1.0/range.x;
		double width = (stereo_mode == PASSIVE_STEREO || stereo_mode == CROSSEYED_STEREO)
			? view_width*0.5 : view_width;
		gcfvec = vector(1.0/range.x, (view_height/width)/range.y, 0.1/range.z);
	} else {
		// TODO: Instead of changing gcf so much, we could change it only when it is 2x
		// off, to aid primitives whose caching may depend on gcf (but are there any?)
		if (gcf != scale) {
			gcf = scale;
			gcf_changed = true;
		}
		gcfvec = vector(gcf,gcf,gcf);
	}
}

void display_kernel::implicit_activate() {
	if (!visible && !explicitly_invisible)
		set_visible( true );
}

void
display_kernel::add_renderable( shared_ptr<renderable> obj)
{
	// Driven from visual/primitives.py set_visible
	if (!obj->translucent()) {
		layer_world.push_back( obj);
	} else
		layer_world_transparent.push_back( obj);
	if (!obj->is_light())
		implicit_activate();
}

void
display_kernel::remove_renderable( shared_ptr<renderable> obj)
{
	// Driven from visual/primitives.py set_visible
	if (!obj->translucent()) {
		std::remove( layer_world.begin(), layer_world.end(), obj);
		layer_world.pop_back();
	}
	else {
		std::remove( layer_world_transparent.begin(), layer_world_transparent.end(), obj);
		layer_world_transparent.pop_back();
	}
}

bool
display_kernel::draw(view& scene_geometry, int whicheye)
{
	// Set up the base modelview and projection matrices
	world_to_view_transform( scene_geometry, whicheye);

	// Render all opaque objects in the world space layer
	enable_lights(scene_geometry);
	world_iterator i( layer_world.begin());
	world_iterator i_end( layer_world.end());
	while (i != i_end) {
		if (i->translucent()) {
			// The color of the object has become transparent when it was not
			// initially.  Move it to the transparent layer.  The penalty for
			// being rendered in the transparent layer when it is opaque is only
			// a small speed hit when it has to be sorted.  Therefore, that case
			// is not tested at all.  (TODO Untrue-- rendering opaque objects in transparent
			// layer makes it possible to have opacity artifacts with a single convex
			// opaque objects, provided other objects in the scene were ONCE transparent)
			layer_world_transparent.push_back( *i.base());
			i = layer_world.erase(i.base());
			continue;
		}
		check_gl_error();
		i->outer_render( scene_geometry);
		check_gl_error();
		++i;
	}

	// Perform a depth sort of the transparent world from back to front.
	if (layer_world_transparent.size() > 1)
		std::stable_sort(
			layer_world_transparent.begin(), layer_world_transparent.end(),
			z_comparator( internal_forward.norm()));

	// Render translucent objects in world space.
	world_trans_iterator j( layer_world_transparent.begin());
	world_trans_iterator j_end( layer_world_transparent.end());
	while (j != j_end) {
		j->outer_render( scene_geometry );
		++j;
	}

	// Render all objects in screen space.
	disable_lights();
	gl_disable depth_test( GL_DEPTH_TEST);
	typedef std::multimap<vector, displaylist, z_comparator>::iterator
		screen_iterator;
	screen_iterator k( scene_geometry.screen_objects.begin());
	screen_iterator k_end( scene_geometry.screen_objects.end());
	while ( k != k_end) {
		//std::cout << "display_kernel.cpp draw calls gl_render" << std::endl;
		k->second.gl_render();
		++k;
	}
	scene_geometry.screen_objects.clear();

	return true;
}


// Renders the entire scene.
bool
display_kernel::render_scene(void)
{
	// TODO: Exception handling?
	if (!realized) {
		realize();
	}

	try {
		recalc_extent();
		view scene_geometry( internal_forward.norm(), center, view_width,
			view_height, forward_changed, gcf, gcfvec, gcf_changed, glext);
		scene_geometry.lod_adjust = lod_adjust;
		scene_geometry.enable_shaders = enable_shaders;
		clear_gl_error();

		on_gl_free.frame();

		glClearColor( background.red, background.green, background.blue, 0);
		// Control which type of stereo to perform.
		switch (stereo_mode) {
			case NO_STEREO:
				scene_geometry.anaglyph = false;
				scene_geometry.coloranaglyph = false;
				glViewport( 0, 0, view_width, view_height);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				draw(scene_geometry, 0);
				break;
			case ACTIVE_STEREO:
				scene_geometry.anaglyph = false;
				scene_geometry.coloranaglyph = false;
				glViewport( 0, 0, view_width, view_height);
				glDrawBuffer( GL_BACK_LEFT);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				draw( scene_geometry, -1);
				glDrawBuffer( GL_BACK_RIGHT);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				draw( scene_geometry, 1);
				break;
			case REDBLUE_STEREO:
				// Red channel
				scene_geometry.anaglyph = true;
				scene_geometry.coloranaglyph = false;
				glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
				glViewport( 0, 0, view_width, view_height);
				glColorMask( GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE);
				draw( scene_geometry, -1);
				// Blue channel
				glColorMask( GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE);
				glClear( GL_DEPTH_BUFFER_BIT);
				draw( scene_geometry, 1);
				// Put everything back
				glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
				break;
			case REDCYAN_STEREO:
				// Red channel
				scene_geometry.anaglyph = true;
				scene_geometry.coloranaglyph = true;
				glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
				glViewport( 0, 0, view_width, view_height);
				glColorMask( GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE);
				draw( scene_geometry, -1);
				// Green and Blue channels
				glColorMask( GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE);
				glClear( GL_DEPTH_BUFFER_BIT);
				draw( scene_geometry, 1);
				// Put everything back
				glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
				break;
			case YELLOWBLUE_STEREO:
				// Red and green channels
				scene_geometry.anaglyph = true;
				scene_geometry.coloranaglyph = true;
				glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
				glViewport( 0, 0, view_width, view_height);
				glColorMask( GL_TRUE, GL_TRUE, GL_FALSE, GL_TRUE);
				draw( scene_geometry, -1);
				// Blue channel
				glColorMask( GL_FALSE, GL_FALSE, GL_TRUE, GL_TRUE);
				glClear( GL_DEPTH_BUFFER_BIT);
				draw( scene_geometry, 1);
				// Put everything back
				glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
				break;
			case GREENMAGENTA_STEREO:
				// Green channel
				scene_geometry.anaglyph = true;
				scene_geometry.coloranaglyph = true;
				glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
				glViewport( 0, 0, view_width, view_height);
				glColorMask( GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
				draw( scene_geometry, -1);
				// Red and blue channels
				glColorMask( GL_TRUE, GL_FALSE, GL_TRUE, GL_TRUE);
				glClear( GL_DEPTH_BUFFER_BIT);
				draw( scene_geometry, 1);
				// Put everything back
				glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
				break;
			case PASSIVE_STEREO: {
				// Also handle viewport modifications.
				scene_geometry.view_width =  view_width/2;
				scene_geometry.anaglyph = false;
				scene_geometry.coloranaglyph = false;
				int stereo_width = int(scene_geometry.view_width);
				// Left eye
				glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
				glViewport( 0, 0, stereo_width, view_height );
				draw( scene_geometry, -1);
				// Right eye
				glViewport( stereo_width+1, 0, stereo_width, view_height);
				draw( scene_geometry, 1);
				break;
			}
			case CROSSEYED_STEREO: {
				// Also handle viewport modifications.
				scene_geometry.view_width =  view_width/2;
				scene_geometry.anaglyph = false;
				scene_geometry.coloranaglyph = false;
				int stereo_width = int(scene_geometry.view_width);
				// Left eye
				glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
				glViewport( 0, 0, stereo_width, view_height);
				draw( scene_geometry, 1);
				// Right eye
				glViewport( stereo_width+1, 0, stereo_width, view_height );
				draw( scene_geometry, -1);
				break;
			}
		}

		// Cleanup
		check_gl_error();
		gcf_changed = false;
		forward_changed = false;
	}

	catch (gl_error e) {
		std::ostringstream msg;
		msg << "render_scene OpenGL error: " << e.what() << ", aborting.\n";
		VPYTHON_CRITICAL_ERROR( msg.str());
		std::exit(1);
	}

	// TODO: Can we delay picking until the Python program actually wants one of these attributes?
	mouse.get_mouse().cam = camera;
	boost::tie( mouse.get_mouse().pick, mouse.get_mouse().pickpos, mouse.get_mouse().position) =
		pick( mouse.get_x(), mouse.get_y() );

	on_gl_free.frame();

	return true;
}

boost::tuple< shared_ptr<renderable>, vector, vector>
display_kernel::pick( int x, int y, float d_pixels)
{
	using boost::scoped_array;

	shared_ptr<renderable> best_pick;
    vector pickpos;
    vector mousepos;
	try {
		clear_gl_error();
		// Notes:
		// culled polygons don't count.  glRasterPos() does count.

		// Allocate a selection buffer of uints.  Format for returned hits is:
		// {uint32: n_names}{uint32: minimunm depth}{uint32: maximum depth}
		// {unit32[n_names]: name_stack}
		// n_names is the depth of the name stack at the time of the hit.
		// minimum and maximum depth are the minimum and maximum values in the
		// depth buffer scaled between 0 and 2^32-1. (source is [0,1])
		// name_stack is the full contents of the name stack at the time of the
		// hit.

		size_t hit_buffer_size = std::max(
				(layer_world.size()+layer_world_transparent.size())*4,
				world_extent.get_select_buffer_depth());
		// Allocate an exception-safe buffer for the GL to talk back to us.
		scoped_array<unsigned int> hit_buffer(
			new unsigned int[hit_buffer_size]);
		// unsigned int hit_buffer[hit_buffer_size];

		// Allocate a std::vector<shared_ptr<renderable> > to lookup names
		// as they are rendered.
		std::vector<shared_ptr<renderable> > name_table;
		// Pass the name stack to OpenGL with glSelectBuffer.
		glSelectBuffer( hit_buffer_size, (GLuint*)hit_buffer.get());
		// Enter selection mode with glRenderMode
		glRenderMode( GL_SELECT);
		glClear( GL_DEPTH_BUFFER_BIT);
		// Clear the name stack with glInitNames(), raise the height of the name
		// stack with glPushName() exactly once.
		glInitNames();
		glPushName(0);

		// Initialize the picking matrix.
		GLint viewport_bounds[4] = {
			0, 0, view_width, view_height
		};
		glMatrixMode( GL_PROJECTION);
		glLoadIdentity();
		gluPickMatrix( (float)x, (float)(view_height - y), d_pixels, d_pixels, viewport_bounds);
		view scene_geometry( internal_forward.norm(), center, view_width, view_height,
			forward_changed, gcf, gcfvec, gcf_changed, glext);
		scene_geometry.lod_adjust = lod_adjust;
		world_to_view_transform( scene_geometry, 0, true);

		// Iterate across the world, rendering each body for picking.
		std::list<shared_ptr<renderable> >::iterator i = layer_world.begin();
		std::list<shared_ptr<renderable> >::iterator i_end = layer_world.end();
		while (i != i_end) {
			glLoadName( name_table.size());
			name_table.push_back( *i);
			{
				(*i)->gl_pick_render( scene_geometry);
			}
			++i;
		}
		std::vector<shared_ptr<renderable> >::iterator j
			= layer_world_transparent.begin();
		std::vector<shared_ptr<renderable> >::iterator j_end
			= layer_world_transparent.end();
		while (j != j_end) {
			glLoadName( name_table.size());
			name_table.push_back( *j);
			{
				(*j)->gl_pick_render( scene_geometry);
			}
			++j;
		}
		// Return the name stack to the bottom with glPopName() exactly once.
		glPopName();

		// Exit selection mode, return to normal rendering rendering. (collects
		// the number of hits at this time).
		size_t n_hits = glRenderMode( GL_RENDER);
		check_gl_error();

		// Lookup the name to get the shared_ptr<renderable> associated with it.
		// The farthest point away in the depth buffer.
		double best_pick_depth = 1.0;
		unsigned int* hit_record = hit_buffer.get();
		unsigned int* const hit_buffer_end = hit_buffer.get() + hit_buffer_size;
		while (n_hits > 0 && hit_record < hit_buffer_end) {
			unsigned int n_names = hit_record[0];
			if (hit_record + 3 + n_names > hit_buffer_end)
				break;
			double min_hit_depth = static_cast<double>(hit_record[1])
				/ 0xffffffffu;
			if (min_hit_depth < best_pick_depth) {
				best_pick_depth = min_hit_depth;
				best_pick = name_table[*(hit_record+3)];
				if (n_names > 1) {
					// Then the picked object is the child of a frame.
					frame* ref_frame = dynamic_cast<frame*>(best_pick.get());
					assert(ref_frame != NULL);
					best_pick = ref_frame->lookup_name(
						hit_record + 4, hit_record + 3 + n_names);
				}
			}
			hit_record += 3 + n_names;
			n_hits--;
		}
		if (hit_record > hit_buffer_end)
			VPYTHON_CRITICAL_ERROR(
				"More objects were picked than could be reported by the GL."
				"  The hit buffer size was too small.");

        tmatrix modelview;
        modelview.gl_modelview_get();
        tmatrix projection;
        projection.gl_projection_get();
        gluUnProject(
            x, view_height - y, best_pick_depth,
            modelview.matrix_addr(),
            projection.matrix_addr(),
            viewport_bounds,
            &pickpos.x, &pickpos.y, &pickpos.z);
        // TODO: Replace the calls to gluUnProject() with own tmatrix inverse
        // and such for optimization
        vector tcenter;
        gluProject( center.x*gcf, center.y*gcf, center.z*gcf,
           	modelview.matrix_addr(),
            projection.matrix_addr(),
            viewport_bounds,
            &tcenter.x, &tcenter.y, &tcenter.z);

        gluUnProject(
        	x, view_height - y, tcenter.z,
        	modelview.matrix_addr(),
        	projection.matrix_addr(),
        	viewport_bounds,
        	&mousepos.x, &mousepos.y, &mousepos.z);
	}
	catch (gl_error e) {
		std::ostringstream msg;
		msg << "pick OpenGL error: " << e.what() << ", aborting.\n";
		VPYTHON_CRITICAL_ERROR( msg.str());
		std::exit(1);
	}
	pickpos.x /= gcfvec.x;
	pickpos.y /= gcfvec.y;
	pickpos.z /= gcfvec.z;
	mousepos.x /= gcfvec.x;
	mousepos.y /= gcfvec.y;
	mousepos.z /= gcfvec.z;
	return boost::make_tuple( best_pick, pickpos, mousepos);
}

void
display_kernel::gl_free()
{
	try {
		clear_gl_error();
		on_gl_free.shutdown();
		check_gl_error();
	}
	catch (gl_error& error) {
		VPYTHON_CRITICAL_ERROR( "Caught OpenGL error during shutdown: "
			+ std::string(error.what())
			+ "; Continuing with the shutdown.");
	}
}

void
display_kernel::allow_spin(bool b)
{
	spin_allowed = b;
}

bool
display_kernel::spin_is_allowed(void) const
{
	return spin_allowed;
}

void
display_kernel::allow_zoom(bool b)
{
	zoom_allowed = b;
}

bool
display_kernel::zoom_is_allowed(void) const
{
	return zoom_allowed;
}

void
display_kernel::set_up( const vector& n_up)
{
	if (n_up == vector())
		throw std::invalid_argument( "Up cannot be zero.");
	vector v = n_up.norm();
	if (v.cross(internal_forward) == vector()) { // if internal_forward parallel to new up, move it away from new up
		if (v.cross(forward) == vector()) {
			// old internal_forward was not parallel to old up
			internal_forward = (forward - 0.0001*up).norm();
		} else {
			internal_forward = forward;
		}
	}
	up = v;
}

shared_vector&
display_kernel::get_up()
{
	return up;
}

void
display_kernel::set_forward( const vector& n_forward)
{
	if (n_forward == vector())
		throw std::invalid_argument( "Forward cannot be zero.");
	vector v = n_forward.norm();
	if (v.cross(up) == vector()) { // if new forward parallel to up, move internal_forward away from up
		// old internal_forward was not parallel to up
		internal_forward = ( v.dot(up)*up + 0.0001*up.cross(internal_forward.cross(up)) ).norm();
	} else { // since new forward not parallel to up, new forward is okay
		internal_forward = v;
	}
	forward = v;
	forward_changed = true;
}

shared_vector&
display_kernel::get_forward()
{
	return forward;
}

void
display_kernel::set_scale( const vector& n_scale)
{
	if (n_scale.x == 0.0 || n_scale.y == 0.0 || n_scale.z == 0.0)
		throw std::invalid_argument(
			"The scale of each axis must be non-zero.");

	vector n_range = vector( 1.0/n_scale.x, 1.0/n_scale.y, 1.0/n_scale.z);
	set_range( n_range );
}

vector
display_kernel::get_scale()
{
	if (autoscale || !range.nonzero())
		throw std::logic_error("Reading .scale and .range is not supported when autoscale is enabled.");
	return vector( 1.0/range.x, 1.0/range.y, 1.0/range.z );
}

void
display_kernel::set_center( const vector& n_center)
{
	center = n_center;
}

shared_vector&
display_kernel::get_center()
{
	return center;
}

void
display_kernel::set_fov( double n_fov)
{
	if (n_fov == 0.0)
		throw std::invalid_argument( "Orthogonal projection is not supported.");
	else if (n_fov < 0.0 || n_fov >= M_PI)
		throw std::invalid_argument(
			"attribute visual.display.fov must be between 0.0 and math.pi "
			"(exclusive)");
	fov = n_fov;
}

double
display_kernel::get_fov()
{
	return fov;
}

void
display_kernel::set_lod(int n_lod)
{
  if (n_lod > 0 || n_lod < -6 )
		throw std::invalid_argument(
		       "attribute visual.display.lod must be between -6 and 0");
  lod_adjust = n_lod;
}

int
display_kernel::get_lod()
{
	return lod_adjust;
}

void
display_kernel::set_uniform( bool n_uniform)
{
	uniform = n_uniform;
}

bool
display_kernel::is_uniform()
{
	return uniform;
}


void
display_kernel::set_background( const rgb& n_background)
{
	background = n_background;
}

rgb
display_kernel::get_background()
{
	return background;
}

void
display_kernel::set_foreground( const rgb& n_foreground)
{
	foreground = n_foreground;
}

rgb
display_kernel::get_foreground()
{
	return foreground;
}

void
display_kernel::set_autoscale( bool n_autoscale)
{
	if (!n_autoscale && autoscale) {
		// Autoscale is disabled, but range_auto remains
		//   set to the current autoscaled scene, until and unless
		//   range is set explicitly.
		recalc_extent();
		range = vector(0,0,0);
	}
	autoscale = n_autoscale;
}

bool
display_kernel::get_autoscale()
{
	return autoscale;
}

bool
display_kernel::get_autocenter()
{
	return autocenter;
}

void
display_kernel::set_autocenter( bool n_autocenter)
{
	autocenter = n_autocenter;
}

void
display_kernel::set_ambient_f( float a)
{
	ambient = rgb( a, a, a);
}

void
display_kernel::set_ambient( const rgb& a)
{
	ambient = a;
}

rgb
display_kernel::get_ambient()
{
	return ambient;
}

void
display_kernel::set_range_d( double r)
{
	set_range( vector(r,r,r) );
}

void
display_kernel::set_range( const vector& n_range)
{
	if (n_range.x == 0.0 || n_range.y == 0.0 || n_range.z == 0.0)
		throw std::invalid_argument(
			"attribute visual.display.range may not be zero.");
	autoscale = false;
	range = n_range;
	range_auto = 0.0;
}

vector
display_kernel::get_range()
{
	if (autoscale || !range.nonzero())
		throw std::logic_error("Reading .scale and .range is not supported when autoscale is enabled.");
	return range;
}

float
display_kernel::get_stereodepth()
{
	return stereodepth;
}

void
display_kernel::set_stereodepth( float n_stereodepth)
{
	if (visible)
		throw std::runtime_error( "Cannot change parameters of an active window");
	else
		stereodepth = n_stereodepth;
}

void
display_kernel::set_stereomode( std::string mode)
{
	if (mode == "nostereo")
		stereo_mode = NO_STEREO;
	else if (mode == "active")
		stereo_mode = ACTIVE_STEREO;
	else if (mode == "passive")
		stereo_mode = PASSIVE_STEREO;
	else if (mode == "crosseyed")
		stereo_mode = CROSSEYED_STEREO;
	else if (mode == "redblue")
		stereo_mode = REDBLUE_STEREO;
	else if (mode == "redcyan")
		stereo_mode = REDCYAN_STEREO;
	else if (mode == "yellowblue")
		stereo_mode = YELLOWBLUE_STEREO;
	else if (mode == "greenmagenta")
		stereo_mode = GREENMAGENTA_STEREO;
	else
		throw std::invalid_argument( "Unimplemented or invalid stereo mode");
}

std::string
display_kernel::get_stereomode()
{
	switch (stereo_mode) {
		case NO_STEREO:
			return "nostereo";
		case ACTIVE_STEREO:
			return "active";
		case PASSIVE_STEREO:
			return "passive";
		case CROSSEYED_STEREO:
			return "crosseyed";
		case REDBLUE_STEREO:
			return "redblue";
		case REDCYAN_STEREO:
			return "redcyan";
		case YELLOWBLUE_STEREO:
			return "yellowblue";
		case GREENMAGENTA_STEREO:
			return "greenmagenta";
		default:
			// Not strictly required, this just silences a warning about control
			// reaching the end of a non-void funciton.
			return "nostereo";
	}
}

std::vector<shared_ptr<renderable> >
display_kernel::get_objects() const
{
	std::vector<shared_ptr<renderable> > ret;
	ret.insert( ret.end(), layer_world.begin(), layer_world.end() );
	ret.insert( ret.end(), layer_world_transparent.begin(), layer_world_transparent.end() );

	// ret[i]->get_children appends the immediate children of ret[i] to ret.  Since
	//   ret.size() keeps increasing, we keep going until we have all the objects in the tree.
	for(size_t i=0; i<ret.size(); i++)
		ret[i]->get_children(ret);

	return ret;
}

std::string
display_kernel::info()
{
	if (!extensions)
		return std::string( "Renderer inactive.\n");
	else {
		std::string s;
		s += "OpenGL renderer active.\n  Vendor: "
		  + vendor
		  + "\n  Version: " + version
		  + "\n  Renderer: " + renderer
		  + "\n  Extensions: ";

		// this->extensions is a list of extensions
		std::ostringstream buffer;
		std::copy( extensions->begin(), extensions->end(),
			std::ostream_iterator<std::string>( buffer, "\n"));
		s += buffer.str();
		return s;
	}
}

void
display_kernel::set_x( float n_x)
{
	if (visible)
		throw std::runtime_error( "Cannot change parameters of an active window");
	else
		window_x = (int)n_x;
}
float
display_kernel::get_x()
{
	return (float)window_x;
}

void
display_kernel::set_y( float n_y)
{
	if (visible)
		throw std::runtime_error( "Cannot change parameters of an active window");
	else
		window_y = (int)n_y;
}
float
display_kernel::get_y()
{
	return (float)window_y;
}

void
display_kernel::set_width( float w)
{
	if (visible)
		throw std::runtime_error( "Cannot change parameters of an active window");
	else
		window_width = (int)w;
}
float
display_kernel::get_width()
{
	return (float)window_width;
}

void
display_kernel::set_height( float h)
{
	if (visible)
		throw std::runtime_error( "Cannot change parameters of an active window");
	else
		window_height = (int)h;
}
float
display_kernel::get_height()
{
	return (float)window_height;
}

void
display_kernel::set_visible( bool vis)
{
	if (!vis) explicitly_invisible = true;
	if (vis != visible) {
		visible = vis;
		set_display_visible( this, visible );
		// drive _activate (through wrap_display_kernel.cpp) in Python code
		activate( vis );
	}
}

bool
display_kernel::get_visible()
{
	return visible;
}

void
display_kernel::set_title( std::string n_title)
{
	if (visible)
		throw std::runtime_error( "Cannot change parameters of an active window");
	else
		title = n_title;
}
std::string
display_kernel::get_title()
{
	return title;
}

bool
display_kernel::is_fullscreen()
{
	return fullscreen;
}
void
display_kernel::set_fullscreen( bool fs)
{
	if (visible)
		throw std::runtime_error( "Cannot change parameters of an active window");
	else
		fullscreen = fs;
}

bool display_kernel::get_exit() { return exit; }
void display_kernel::set_exit(bool b) { exit = b; }

bool
display_kernel::is_showing_toolbar()
{
	return show_toolbar;
}

void
display_kernel::set_show_toolbar( bool fs)
{
	if (visible)
		throw std::runtime_error( "Cannot change parameters of an active window");
	show_toolbar = fs;
}

mouse_t*
display_kernel::get_mouse()
{
	implicit_activate();
	return &mouse.get_mouse();
}

void
display_kernel::set_selected( shared_ptr<display_kernel> d )
{
	selected = d;
}

shared_ptr<display_kernel>
display_kernel::get_selected()
{
	return selected;
}

bool
display_kernel::hasExtension( const std::string& ext ) {
	return extensions->find( ext ) != extensions->end();
}

// The small platform-specific getProcAddress functions are in the platform-specific font_renderer files.
/*
display_kernel::EXTENSION_FUNCTION
display_kernel::getProcAddress( const char* x ) {
	if ( !strcmp(x, "display_kernel::getProcAddress" ) ) return notImplemented;
	return NULL;
}
*/

} // !namespace cvisual
