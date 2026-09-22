from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extra_compile_args = ['-O3', '-std=c++17', '-fPIC']

ext_c_ext = Extension(
    'c_extensions.fast_mwc',
    sources=[
        'c_extensions/fast_mwc.pyx',
        'c_extensions/mwc_kernel.cpp',
    ],
    include_dirs=[
        np.get_include(),
        'c_extensions',
        '.',
    ],
    language='c++',
    extra_compile_args=extra_compile_args,
)

setup(
    name='fast_mwc_pkg',
    ext_modules=cythonize(
        [ext_c_ext],
        language_level='3',
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    )
)
