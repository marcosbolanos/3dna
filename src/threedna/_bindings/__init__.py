"""
Geometric bindings - C++ extensions compiled from cpp/ directory.

These are pybind11 bindings wrapping geometry-central for exact geodesics
and libigl for surface projections.

To rebuild:
    cd ../cpp/build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/path/to/venv/python
    make -j4
    cp *.so ../_bindings/
"""

import os

# Add this directory to path so Python can find the .so files
_bindings_dir = os.path.dirname(__file__)
