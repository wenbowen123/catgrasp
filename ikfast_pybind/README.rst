==============
ikfast_pybind
==============

.. start-badges

.. image:: https://travis-ci.com/yijiangh/ikfast_pybind.svg?branch=master
    :target: https://travis-ci.com/yijiangh/ikfast_pybind
    :alt: Travis CI

.. image:: https://img.shields.io/github/license/yijiangh/conmech
    :target: ./LICENSE
    :alt: License MIT

.. image:: https://img.shields.io/badge/python-2.5+|3.x-blue
    :target: https://pypi.org/project/ikfast_pybind/
    :alt: PyPI - Python Version

.. image:: https://img.shields.io/badge/pypi-v0.0.1-orange
    :target: https://pypi.org/project/ikfast_pybind/
    :alt: PyPI - Latest Release


**ikfast_pybind** is a python binding generation library for the analytic kinematics engine `IKfast <http://openrave.org/docs/1.8.2/openravepy/ikfast/>`__. 
The python bindings are generated via `pybind11 <https://github.com/pybind/pybind11>`_ a `CMake <https://cmake.org/>`_-based build system.

**Note:** You need the ikfast `.h` and `.cpp` ready to generate the python bindings. This *URDF-to-cpp* generation part needs to be done with `openrave` and **IS NOT** done by this repo, 
please see `this tutorial <http://docs.ros.org/kinetic/api/framefab_irb6600_support/html/doc/ikfast_tutorial.html>`_ for details.

The assembly sequence and motion planning framework `pychoreo <https://github.com/yijiangh/pychoreo>`_ 
relies on this library to generate compatible IK modules for robots across brands, scales, and dofs.

Prerequisites
-------------

*ikfast_pybind* depends on the following dependencies, which come from pybind11 for building the python bindings.

**On Unix (Linux, OS X)**

* A compiler with C++11 support
* CMake >= 2.8.12

**On Windows**

* Visual Studio 2015 (required for all Python versions, see notes below)
* CMake >= 3.1

**It is recommended (especially for Windows users) to test the environment with the**
`cmake_example for pybind11 <https://github.com/pybind/cmake_example>`_ **before proceeding to build conmech.**

Installation
------------

::

  git clone --recursive https://github.com/yijiangh/ikfast_pybind
  cd ikfast_pybind
  pip install .
  # try with '--user' if you encountered a sudo problem

For developers:

::

  git clone --recursive https://github.com/yijiangh/ikfast_pybind
  cd ikfast_pybind
  python setup.py sdist
  pip install --verbose dist/*.tar.gz

With the ``setup.py`` file included in the base folder, the pip install command will invoke CMake and build the pybind11 module as specified in CMakeLists.txt.

References
----------

Citation
^^^^^^^^

If you find `IKFast <http://openrave.org/docs/0.8.2/openravepy/ikfast/>`__ useful, 
please cite `OpenRave <http://openrave.org/>`_:

::

  @phdthesis{diankov_thesis,
    author = "Rosen Diankov",
    title = "Automated Construction of Robotic Manipulation Programs",
    school = "Carnegie Mellon University, Robotics Institute",
    month = "August",
    year = "2010",
    number= "CMU-RI-TR-10-29",
    url={http://www.programmingvision.com/rosen_diankov_thesis.pdf},
  }

Related links
^^^^^^^^^^^^^

`tutorial on ikfast cpp generation <http://docs.ros.org/kinetic/api/framefab_irb6600_support/html/doc/ikfast_tutorial.html>`_: See this tutorial for a detailed instruction on how to generate the ikfast cpp code from an URDF.

`Testing ikfast modules with a pick-n-place demo in pybullet <https://github.com/yijiangh/conrob_pybullet/tree/master/debug_examples>`_

pybind11_
