#!/bin/bash
#
# Helper script to grab all the packages vpython needs Run as root.
#
apt-get -q -y update
apt-get -q -y install git
apt-get -q -y install libgtk2.0-dev
apt-get -q -y install libgtkglextmm-x11-1.2-dev
apt-get -q -y install libgtkmm-2.4-dev
apt-get -q -y install python-dev
apt-get -q -y install python-setuptools
apt-get -q -y install python-numpy
apt-get -q -y install libboost-python-dev
apt-get -q -y install libboost-signals-dev
apt-get -q -y install libghc-gstreamer-dev
apt-get -q -y install python-tk

