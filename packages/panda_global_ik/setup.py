import os

from setuptools import Extension, setup


def eigen_include() -> str:
    candidates = [os.environ.get('EIGEN_INCLUDE'), os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include'),
                  '/usr/include', '/usr/local/include']
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, 'eigen3')):
            return c
    raise RuntimeError('Eigen 3 not found: `sudo apt install libeigen3-dev` (or `conda install -c conda-forge eigen`), '
                       'or set EIGEN_INCLUDE to the folder that contains eigen3/')


# Loaded with ctypes, not imported: the shared library only needs the C entry points of src/geofik_c.cpp.
setup(ext_modules=[Extension(
    'panda_global_ik._geofik',
    sources=['third_party/geofik/geofik.cpp', 'src/geofik_c.cpp'],
    include_dirs=[eigen_include(), 'third_party/geofik'],
    language='c++',
    extra_compile_args=['-O3', '-std=c++17'],
)])
