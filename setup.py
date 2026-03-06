import os
import shutil

from setuptools import setup

ext_modules = []
cmdclass = {}

enable_cpp = os.environ.get("THREEDNA_ENABLE_CPP_BACKEND", "0") == "1"
has_compiler = any(shutil.which(name) is not None for name in ("c++", "g++", "clang++"))

if enable_cpp and has_compiler:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "threedna._surface_kernels_cpp",
            ["src/threedna/cpp/surface_kernels.cpp"],
            cxx_std=17,
        ),
    ]
    cmdclass = {"build_ext": build_ext}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
