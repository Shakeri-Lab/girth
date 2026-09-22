from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extra_compile_args = ["-O3", "-std=c++17", "-fPIC"]

ext = Extension(
    "fast_mwc",
    sources=[
        "fast_mwc.pyx",
        "mwc_kernel.cpp",
    ],
    include_dirs=[
        np.get_include(),
        ".",
    ],
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="fast_mwc_pkg",
    ext_modules=cythonize(
        [ext],
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        }
    )
)
