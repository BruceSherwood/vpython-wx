#
# script to actually build the package.
#

PKG_MAKER="/Developer-3.2.5/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker"
PM_DOC="VPY-6.0-beta-py27.pmdoc"
PKG_OUT="./VisualPythonInstaller/VPython-Complete-6.01B.pkg"

$PKG_MAKER --verbose --doc $PM_DOC --out $PKG_OUT
