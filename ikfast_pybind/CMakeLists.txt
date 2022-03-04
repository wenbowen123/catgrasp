cmake_minimum_required(VERSION 2.8.12)
project(ikfast_pybind)

set(IKFAST_PYBIND_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(IKFAST_PYBIND_SOURCE_DIR "${IKFAST_PYBIND_ROOT}/src")
set(IKFAST_PYBIND_EXTERNAL "${IKFAST_PYBIND_ROOT}/ext")

add_subdirectory(${IKFAST_PYBIND_EXTERNAL})
add_subdirectory(${IKFAST_PYBIND_SOURCE_DIR})
