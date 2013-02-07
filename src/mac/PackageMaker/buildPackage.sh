#
# script to actually build the package.
#

VERSION="6.02"
HOME_DIR=~
PKG_MAKER_DIR="src/mac/PackageMaker"
PKG_MAKER="/Developer-3.2.5/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker"
PM_DOC="$PKG_MAKER_DIR/VPY-6.0-beta-py27.pmdoc"
PKG_DIR="$PKG_MAKER_DIR/VisualPythonInstaller"
PKG_OUT="$PKG_DIR/VPython-Complete-$VERSION$B.pkg"
DMG_OUT="$HOME_DIR/Dropbox/Public/VPython-$VERSION$-Py2.7-10.6.dmg"

if [ "$1" == "pkg" ]; then

    if [ ! -e $PKG_DIR ]; then
	mkdir -p $PKG_DIR
    fi

    if [ -e $PKG_OUT ]; then
	echo "deleting old package"
	rm $PKG_OUT
    fi

    echo $PKG_MAKER --verbose --doc $PM_DOC --out $PKG_OUT
    $PKG_MAKER --verbose --doc $PM_DOC --out $PKG_OUT
fi

if [ "$1" == "dmg" ]; then

    if [ -e $DMG_OUT ]
    then
	echo "deleting old image"
	rm $DMG_OUT
    fi

    echo hdiutil create -volname \"VPython_Installer\" -fs HFS+ -srcfolder $PKG_DIR $DMG_OUT
    hdiutil create -volname \"VPython_Installer\" -fs HFS+ -srcfolder $PKG_DIR $DMG_OUT
fi
