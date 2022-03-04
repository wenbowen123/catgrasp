from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PG_OP',
    ext_modules=[
        CUDAExtension('PG_OP', [
            'src/pointgroup_ops_api.cpp',

            'src/pointgroup_ops.cpp',
            'src/cuda.cu',
        ],
        include_dirs=['eigen3/'],
        extra_compile_args={'cxx': ['-std=c++14','-O3'], 'nvcc': ['-O3']})
    ],
    cmdclass={'build_ext': BuildExtension}
)