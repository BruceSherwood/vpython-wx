#!/bin/sh
#
# autogen.sh 
#
# Requires: automake, autoconf, libtool

aclocal
libtoolize
automake --foreign --add-missing
autoconf

# Verify that everything was generated correctly.  Note that some of these
# may just be symlinks, which is OK.

if [ -e Makefile.in ] ; then 
    echo "ok Makefile.in"
else
    echo "Makefile.in not generated"
fi

if [ -e aclocal.m4 ] ; then 
    echo "ok aclocal.m4"
else
    echo "aclocal.m4n not generated"
fi

if [ -e config.guess ] ; then 
    echo "ok config.guess"
else
    echo "config.guess not generated"
fi

if [ -e config.sub ] ; then 
    echo "ok config.sub"
else
    echo "config.sub not generated"
fi

if [ -e configure ] ; then 
    echo "ok configure"
else
    echo "configure not generated"
fi

if [ -e install-sh ] ; then 
    echo "ok install-sh"
else
    echo "install-sh not generated"
fi

if [ -e ltmain.sh ] ; then 
    echo "ok ltmain.sh"
else
    echo "ltmain.sh not generated"
fi

if [ -e missing ] ; then 
    echo "ok missing"
else
    echo "missing not generated"
fi

if [ -e py-compile ] ; then 
    echo "ok py-compile"
else
    echo "py-compile not generated"
fi

if [ -e site-packages/visual/Makefile.in ] ; then 
    echo "ok site-packages/visual/Makefile.in"
else
    echo "site-packages/visual/Makefile.in not generated"
fi

if [ -e site-packages/vis/Makefile.in ] ; then 
    echo "ok site-packages/vis/Makefile.in"
else
    echo "site-packages/vis/Makefile.in not generated"
fi

if [ -e examples/Makefile.in ] ; then 
    echo "ok examples/Makefile.in"
else
    echo "examples/Makefile.in not generated"
fi

if [ -e docs/Makefile.in ] ; then 
    echo "ok docs/Makefile.in"
else
    echo "docs/Makefile.in not generated"
fi

if [ -e Makefile.in ] && [ -e aclocal.m4 ] && [ -e config.guess ] \
	&& [ -e config.sub ] && [ -e configure ] && [ -e install-sh ] \
	&& [ -e ltmain.sh ] && [ -e missing ] && [ -e py-compile ] \
	&& [ -e site-packages/visual/Makefile.in ] \
	&& [ -e site-packages/vis/Makefile.in ] \
	&& [ -e examples/Makefile.in ] && [ -e docs/Makefile.in ] ; then
	echo "Completed successfully"
else
	echo "One or more generated files was not created properly."
	exit 1
fi

exit 0
