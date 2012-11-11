## Process this file with automake to produce Makefile.in
# Makefile to distribute the documentation

AUTOMAKE_OPTIONS = foreign

srcdir = @srcdir@
VPATH = @srcdir@

if BUILD_DOCS
docfiles = index.html \
	visual/arrow.html visual/axes.gif visual/box.html visual/color.html \
	visual/cone.html visual/controls.html visual/convex.html visual/curve.html \
	visual/cylinder.html visual/defaults.html visual/delete.html visual/display.html \
	visual/ellipsoid.html visual/extrusion_overview.py visual/extrusion.html \
	visual/faces.html visual/factorial.html \
	visual/files.html visual/float.html \
	visual/frame.html visual/graph.html visual/helix.html visual/index.html \
	visual/keyboard.html visual/label.html visual/license.txt visual/lights.html \
	visual/materials.html visual/mouse.html visual/mouse_click.html \
	visual/mouse_drag.html visual/new_features.html visual/options.html \
	visual/paths.html visual/points.html visual/primitives.html \
	visual/pyramid.html visual/rate.html visual/ring.gif visual/ring.html \
	visual/rotation.html visual/shapes.html visual/sphere.html visual/spherebox.gif \
	visual/text.html visual/trail.html visual/vector.html \
	visual/VisualIntro.html visual/VisualRef.css \
	visual/VisualRef-1.gif visual/VisualRef-2.gif visual/VisualRef-3.gif \
	visual/VisualRef-4.gif visual/VisualRef-6.gif visual/VisualRef-7.gif \
	visual/VPython_Intro.pdf \
	visual/images/arc_and_triangle_with_hole.jpg visual/images/arc_shape.jpg \
	visual/images/arc.jpg \
	visual/images/arrow.jpg visual/images/box.jpg visual/images/circle_frame.jpg \
	visual/images/circle_shape.jpg visual/images/circle.jpg visual/images/colors_abutting.jpg \
	visual/images/cone.jpg visual/images/constructive_geometry.jpg \
	visual/images/controls.jpg visual/images/convex.jpg visual/images/cross_ellipse.jpg \
	visual/images/cross_shape.jpg visual/images/cross.jpg visual/images/curve.jpg \
	visual/images/cylinder.jpg visual/images/ellipse_shape.jpg visual/images/ellipse.jpg \
	visual/images/ellipsoid.jpg visual/images/extrusion_twist.jpg \
	visual/images/extrusion.jpg visual/images/extrustion_scaling.jpg \
	visual/images/faces.jpg visual/images/filedialog.jpg visual/images/frustum.jpg \
	visual/images/gear.jpg visual/images/graph.jpg visual/images/helix.jpg \
	visual/images/heptagon.jpg visual/images/hexagon_shape.jpg visual/images/hexagon.jpg \
	visual/images/label.jpg visual/images/lathe.jpg \
	visual/images/material_etc.jpg visual/images/ngon_shape.jpg \
	visual/images/pentagon_shape.jpg visual/images/pentagon.jpg \
	visual/images/rackgear.jpg \
	visual/images/points.jpg visual/images/pyramid.jpg visual/images/rectangle_chamfered.jpg \
	visual/images/rectangle_rotated.jpg visual/images/rectangle_rounded.jpg \
	visual/images/rectangle_shape_chamfered.jpg \
	visual/images/rectangle_shape_frame.jpg visual/images/rectangle_shape_rotated.jpg \
	visual/images/rectangle_shape_rounded.jpg \
	visual/images/rectangle_shape.jpg visual/images/rectangle.jpg \
	visual/images/ring.jpg visual/images/skin.jpg \
	visual/images/sphere.jpg visual/images/star_shape.jpg visual/images/star_with_hole.jpg \
	visual/images/star.jpg visual/images/starpath.jpg visual/images/text_extrusion.jpg \
	visual/images/text_inverse.jpg visual/images/text_shape.jpg \
	visual/images/text.jpg visual/images/text3d.jpg \
	visual/images/trail.jpg \
	visual/images/trapezoid_shape.jpg visual/images/trapezoid.jpg \
	visual/images/triangle_circular_hole.jpg visual/images/triangle_frame.jpg \
	visual/images/triangle_path.jpg visual/images/triangle_shape.jpg \
	visual/images/triangle_with_hole.jpg visual/images/triangle_with_hole2.jpg \
	visual/images/triangle.jpg visual/images/triangular_ring.jpg
else
docfiles = 
endif

pkgdatadir = @visualdocdir@

nobase_dist_pkgdata_DATA = $(docfiles)
