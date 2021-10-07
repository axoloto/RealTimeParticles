
cmake_minimum_required(VERSION 3.12)
project(RealTimeParticles VERSION 1.0
                          DESCRIPTION "Minimalist real-time 3D particle system simulator"
                          HOMEPAGE_URL "https://github.com/axoloto/RealTimeParticles")

message(STATUS "Version Major ${PROJECT_VERSION_MAJOR}")
message(STATUS "Version Minor ${PROJECT_VERSION_MINOR}")
message(STATUS "Version Patch ${PROJECT_VERSION_PATCH}")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_compile_definitions(DEBUG_BUILD)
endif()

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

# Packaging support
set(CPACK_PACKAGE_VENDOR "Adrien Moulin")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Minimalist real-time 3D particles system simulator")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_DIR})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README_Installer.txt")

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" DESTINATION .)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README_Installer.txt" DESTINATION .)

include(CPack)