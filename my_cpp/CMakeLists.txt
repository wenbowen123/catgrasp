cmake_minimum_required(VERSION 2.8.10)
project(my_cpp)


set(CMAKE_BUILD_TYPE Release)
add_compile_options(-std=c++14 -fopenmp)

find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(OpenMP REQUIRED)
find_package(FCL REQUIRED)
find_package(pybind11 REQUIRED)

set(PYBIND11_FINDPYTHON ON)


include_directories(
  ${FCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/../ikfast_pybind/src
)


file(GLOB MY_SRC "${CMAKE_SOURCE_DIR}/*.cpp")
list(REMOVE_ITEM MY_SRC "${CMAKE_SOURCE_DIR}/pybind.cpp")

add_library(MY_LIB SHARED ${MY_SRC} "${CMAKE_SOURCE_DIR}/../ikfast_pybind/src/kuka_iiwa14/ikfast0x1000004a.Transform6D.0_1_3_4_5_6_f2.cpp")
target_link_libraries(MY_LIB ${FCL_LIBRARIES} ${OpenMP_CXX_FLAGS})

pybind11_add_module(my_cpp pybind.cpp)
target_link_libraries(my_cpp PRIVATE MY_LIB)
