from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name = 'fast_utils',
    ext_modules = cythonize([
    Extension("fast_utils", 
              ["fast_utils.pyx"], 
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp']
    )
    ])
)
