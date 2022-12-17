
file(GLOB SRC "*.cpp" "*.hpp")

find_package(spdlog REQUIRED CONFIG)

add_library(utils ${SRC})

target_compile_definitions(utils PUBLIC -DSOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_definitions(utils PUBLIC -DINSTALL_DIR="${CMAKE_INSTALL_DIR}")
target_compile_definitions(utils PUBLIC -DVERSION_MAJOR="${PROJECT_VERSION_MAJOR}")
target_compile_definitions(utils PUBLIC -DVERSION_MINOR="${PROJECT_VERSION_MINOR}")
target_compile_definitions(utils PUBLIC -DVERSION_PATCH="${PROJECT_VERSION_PATCH}")
message(STATUS "CMAKE Source Directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE Install Directory: ${CMAKE_INSTALL_DIR}")

target_link_libraries(utils PUBLIC spdlog::spdlog)

target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS utils DESTINATION lib)

