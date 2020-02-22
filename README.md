# GigPy 
[![Language](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPL-yellow.svg)](https://github.com/ymcmrs/GigPy/blob/master/LICENSE)

Gps-based Imaging Geodesy toolbox in PYthon (GigPy) is an open source package for analyzing spaito-temporal GPS products (tropospheric delays, water vapor delay or atmospheric water vapor, ground displacement) from an imaging-geodesy perspective. It searches and downloads gps products over an area of interest from [Nevada Geodetic Laboratory](http://geodesy.unr.edu/) or [UNAVCO](https://www.unavco.org/), uses state-of-the art algorithms to analyze different components (topography-correlated, spatial trend component, turbulent component) of the GPS products, and robustly reconstructs the high-resolution maps of different GPS products, which could be useful in areas of meteorology (atmospheric water vapor maps) and geodesy (tropospheric delay maps). Performance of GigPy definitely depends on the spatial density of the available GPS stations, and according to our test-purpose experiments, it works well in areas like Northern America, Europe, and Japan.  

ps: GigPy is Improved from PyGPS.
