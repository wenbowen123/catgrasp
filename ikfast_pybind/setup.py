import os
import re
import sys
import platform
import subprocess
import io

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from setup_cmake_utils import CMakeExtension, CMakeBuild


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def read(*names, **kwargs):
    return io.open(
        os.path.join(ROOT_DIR, *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


long_description = read('README.rst')

requirements = [
    'cmake>=3.18',
]

EXT_MODULES = [
               CMakeExtension('ikfast_kuka_iiwa14'),
               ]

setup(
    name='ikfast_pybind',
    version='0.1.1',
    license='MIT License',
    description='ikfast_pybind is a python binding generation library for the analytic kinematics engine ikfast.',
    author='Yijiang Huang',
    author_email='yijiangh@mit.edu',
    url="https://github.com/yijiangh/ikfast_pybind",
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M |
                   re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    # packages=['ikfast_pybind'],
    # package_dir={'': 'src'},
    ext_modules=EXT_MODULES,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        "License :: OSI Approved :: MIT License",
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=['Robotics', 'kinematics'],
    install_requires=requirements,
)
