# Repo tools

- use `uv` for adding packages and running commands
- Dev dependencies are added with `uv add --dev`
- Project is installed as editable for testing
- Typecheck with `uv run pyright`
- Lint with `uv run ruff`

# Building C++ bindings

```bash
cd src/threedna/cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

Dependencies (Eigen, libigl, geometry-central, TinyAD) are fetched via FetchContent.

### Build with non-system clang (e.g., brew on Linux)

If using clang from Homebrew on Linux (not system-installed), set the standard library explicitly:

```bash
LDFLAGS="-stdlib=libc++ -L/home/linuxbrew/.linuxbrew/lib" CXXFLAGS="-stdlib=libc++" \
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
```

## Conventions and rules

- never use dynamic imports. Every library used must be explicitly in the project's dependencies. Dev dependencies are separate.
