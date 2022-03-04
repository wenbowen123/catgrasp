from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools import setup, Extension, find_packages
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../ikfast_pybind')
from setup_cmake_utils import CMakeExtension, CMakeBuild
from pybind11.setup_helpers import Pybind11Extension


class PostDevelopCmd(develop):
    def run(self):
        develop.run(self)

class PostInstallCmd(install):
    def run(self):
        install.run(self)


ext_modules = [CMakeExtension('my_cpp'),
               ]


setup(name='my_cpp',
    version='0.0.0',
    author='Bowen Wen',
    author_email='wenbowenxjtu@gmail.com',
    ext_modules=ext_modules,
    cmdclass={
        'install': PostInstallCmd,
        'develop': PostDevelopCmd,
        "build_ext": CMakeBuild,
    }
)
