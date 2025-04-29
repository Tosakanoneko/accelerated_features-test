from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("model_test", language_level=3)
)