# Geometric Bindings

C++ extensions compiled from the `../cpp/` directory using pybind11.

## Files

- `_geodesic_cpp.so` — Exact geodesic computation via geometry-central (MMP algorithm)
- `_surface_kernels_cpp.so` — Surface projection via libigl

## Building

```bash
cd ../cpp/build

# On Linux with brew clang:
LDFLAGS="-stdlib=libc++ -L/home/linuxbrew/.linuxbrew/lib" CXXFLAGS="-stdlib=libc++" \
  cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/path/to/venv/python

# Or with system compiler:
cmake .. -DCMAKE_BUILD_TYPE=Release

make -j4
```

## Requirements

- CMake ≥ 3.16
- C++17 compiler (GCC, Clang, or MSVC)
- Python 3.7+ (must match your Python environment)
- Dependencies fetched via CMake FetchContent: Eigen, libigl, geometry-central, pybind11

---

## For Developers

### Fast iteration (after first build)

After the initial build, `_deps/` contains compiled dependencies. For quick iteration:

```bash
cd ../cpp/build
make bindings -j4
```

This only recompiles the binding `.cpp` files (~5 seconds), skipping deps.

### Full rebuild

```bash
cd ../cpp/build
rm -rf *  # WARNING: also deletes _deps/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### First build takes long

The first build downloads and compiles geometry-central (~2 min). Subsequent builds are fast.
