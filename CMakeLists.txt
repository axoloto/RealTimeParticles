
cmake_minimum_required(VERSION 3.12)
project(boids VERSION 0.1)

message(STATUS "version 0.1")
message(STATUS "build type ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(OpenGL REQUIRED)

if(USE_OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARIES})
    add_definitions("-DOPENCL_ACTIVATED")
    message ("-- OpenCL include => ${OpenCL_INCLUDE_DIRS}")
    message ("-- OpenCL library => ${OpenCL_LIBRARIES}")
    message ("-- OpenCL version => ${OpenCL_VERSION_STRING}")
endif()

include_directories("third_parties")
include_directories("physics")
include_directories("render")
include_directories("ui")
include_directories("app")

add_subdirectory("third_parties")
add_subdirectory("physics")
add_subdirectory("render")
add_subdirectory("ui")
add_subdirectory("app")