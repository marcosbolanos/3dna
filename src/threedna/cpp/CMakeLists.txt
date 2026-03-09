cmake_minimum_required(VERSION 3.16)

# For old CMakeLists.txt in fetched dependencies
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Auto-detect libc++ from brew on Linux (must set BEFORE project())
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(EXISTS "/home/linuxbrew/.linuxbrew/lib/libc++.so")
    set(CMAKE_CXX_FLAGS "-stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "-stdlib=libc++ -L/home/linuxbrew/.linuxbrew/lib")
    set(CMAKE_SHARED_LINKER_FLAGS "-stdlib=libc++ -L/home/linuxbrew/.linuxbrew/lib")
  endif()
endif()

project(threedna_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Eigen (header-only)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# libigl (header-only, includes predicates)
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.4.0
)
set(LIBIGL_WITH_OPENGL OFF)
set(LIBIGL_WITH_OPENGL_GLFW OFF)
set(LIBIGL_WITH_EMBREE OFF)
set(LIBIGL_WITH_TETGEN OFF)
set(LIBIGL_WITH_TRIANGLE OFF)
set(LIBIGL_PREDICATES ON)
FetchContent_MakeAvailable(libigl)

# geometry-central
FetchContent_Declare(
  geometrycentral
  GIT_REPOSITORY https://github.com/nmwsharp/geometry-central.git
  GIT_TAG 4f5fe7e
)
FetchContent_MakeAvailable(geometrycentral)

# TinyAD
FetchContent_Declare(
  tinyad
  GIT_REPOSITORY https://github.com/patr-schm/tinyad.git
  GIT_TAG 75093e14ef0d7bb39657c5f3b2aba1251afaa38c
)
FetchContent_MakeAvailable(tinyad)

# pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Add your binding sources
add_library(threedna_cpp MODULE
  surface_kernels.cpp
  # Add more .cpp files here as you create them
)

target_link_libraries(threedna_cpp PRIVATE
  pybind11::module
  geometry-central
  igl::core
  igl::predicates
  TinyAD
  Eigen3::Eigen
)

target_include_directories(threedna_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test executable for geodesics
add_executable(test_geodesic test_geodesic.cpp)
target_link_libraries(test_geodesic PRIVATE
  geometry-central
  Eigen3::Eigen
)
target_compile_features(test_geodesic PRIVATE cxx_std_17)
