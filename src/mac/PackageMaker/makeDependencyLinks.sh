#
# after a fresh checkout from git, use this to point dependencies back to boost
# I really should change this in setup.py to point there directly. Not today!
# too many other changes to make.
#

mkdir -p dependencies/boost_files
ln -s ~/Development/boost_1_52_0/boost ./dependencies/boost_files/boost
ln -s ~/Development/boost_1_52_0/stage/lib ./dependencies/boost_files/mac_libs
