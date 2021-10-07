
file(GLOB SRC "*.cpp" "*.hpp")

add_library(utils ${SRC})

add_definitions(-DSOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_definitions(-DINSTALL_DIR="${CMAKE_INSTALL_DIR}")
add_definitions(-DVERSION_MAJOR="${PROJECT_VERSION_MAJOR}")
add_definitions(-DVERSION_MINOR="${PROJECT_VERSION_MINOR}")
message(STATUS "CMAKE Source Directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE Install Directory: ${CMAKE_INSTALL_DIR}")

install(TARGETS utils DESTINATION lib)

