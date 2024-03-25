from distutils.core import setup
from Cython.Build import cythonize

setup(
	ext_modules=cythonize(["gol_cy1.pyx", "gol_cy2.pyx", "gol_cy3.pyx", "gol_cy4.pyx"])
	)