
file(GLOB SRC "*.cpp" "*.hpp")

add_library (utils ${SRC})

add_definitions(-DBINARY_DIR="${CMAKE_SOURCE_DIR}")
message(STATUS "${CMAKE_SOURCE_DIR}")

install(TARGETS utils RUNTIME DESTINATION bin LIBRARY DESTINATION lib)

