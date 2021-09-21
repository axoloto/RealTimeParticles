
cmake_minimum_required(VERSION 3.12)
project(boids VERSION 0.1)

message(STATUS "version 0.1")
add_definitions(-DBUILD_TYPE="${CMAKE_BUILD_TYPE}")
message(STATUS "build type ${CMAKE_BUILD_TYPE}")
message(STATUS "build dir ${CMAKE_BUILD_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(${CMAKE_BUILD_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(OpenGL REQUIRED)

include_directories("utils")
include_directories("physics")
include_directories("render")
include_directories("ui")
include_directories("app")

add_subdirectory("utils")
add_subdirectory("physics")
add_subdirectory("render")
add_subdirectory("ui")
add_subdirectory("app")