
cmake_minimum_required(VERSION 3.12)
project(boids VERSION 0.1)

message(STATUS "version 0.1")
message(STATUS "build type ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include_directories("app")
include_directories("core")
include_directories("third_parties")

add_subdirectory("app")
add_subdirectory("core")
