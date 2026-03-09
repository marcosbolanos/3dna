# threedna
Efficient generation of DNA origami 3D shapes

## Building the C++ Bindings

```bash
cd src/threedna/cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

This builds the pybind11 module with dependencies fetched via FetchContent (Eigen, libigl, geometry-central, TinyAD).
