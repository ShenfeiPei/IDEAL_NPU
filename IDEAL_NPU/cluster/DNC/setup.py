import os
import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('DNC', parent_package, top_path)
    config.add_extension('DNC_',
                         sources=['DNC_.pyx'],
                         include_dirs=[numpy.get_include()],
                         language="c++",
                         libraries=libraries)

    config.ext_modules = cythonize(config.ext_modules, nthreads=4, compiler_directives={'language_level': 3})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
