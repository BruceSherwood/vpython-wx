#!/bin/bash -
#title       : vpythonlinuxinstall.sh
#description : A shell script to install VPython on Linux 
#author      : Aishwarya Unnikrishnan (github.com/katanachan)
#email id    : shwarya.unnikrishnan@gmail.com
#usage       : bash vpythonlinuxinstall.sh or ./vpythonlinuxinstall.sh
#=====================================================================   

sudo apt-get install libwxgtk3.0-dev
sudo apt-get install virtualenv
wget http://downloads.sourceforge.net/wxpython/wxPython-src-3.0.2.0.tar.bz2
tar -xvjf wxPython-src-3.0.2.0.tar.bz2
mkvirtualenv workspace	
workon workspace
cd wxPython-src-3.0.2.0/
./configure --prefix=/home/$SUDO_USER/.virtualenvs/workspace/lib/ --with-gtk2 --enable-unicode --with-opengl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$SUDO_USER/.virtualenvs/workspace/lib
cd wxPython
CFLAGS=-Wno-error=format-security CPPFLAGS=-Wno-error=format-security sudo python setup.py install
python setup.py build
cd ../..
deactivate 

wget https://pypi.python.org/packages/dd/69/a6d5ba016f4e15a83e49471bcf91a7b8fbdf818e5acb002f554027d47650/TTFQuery-1.0.5.tar.gz
tar -xvjf TTFQuery-1.0.5.tar.gz
cd TTFQuery-1.0.5/
python setup.py build && sudo python setup.py install
cd ..

sudo pip install fonttools
 
wget https://pypi.python.org/packages/73/9e/fe761e03de28b51b445ddf01ddae87441b7e7040df7d830b86db8f945808/Polygon2-2.0.8.tar.gz#md5=3349a6dfc4cda2a1bcc9bf6c9d411470
tar -xvjf Polygon2-2.0.8.tar.gz
cd Polygon2-2.0.8/
python setup.py build && sudo python setup.py install
cd ..

sudo apt-get update
sudo apt-get install git
sudo apt-get install libgtk2.0-dev
sudo apt-get install libgtkglextmm-x11-1.2-dev
sudo apt-get install libgtkmm-2.4-dev
sudo apt-get install python-dev
sudo apt-get install python-setuptools
sudo apt-get install python-numpy
sudo apt-get install libboost-python-dev
sudo apt-get install libboost-signals-dev
sudo apt-get install tk

wget http://sourceforge.net/projects/vpythonwx/files/6.11-release/vpython-wx-src.6.11.tgz/download
cd vpython-wx-src.6.11/
python setup.py build && sudo python setup.py install
