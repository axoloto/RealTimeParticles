
file(GLOB SRC "*.cpp" "*.hpp")

add_library(render ${SRC})

target_include_directories(render PRIVATE ${OPENGL_INCLUDE_DIRS})

install(TARGETS render DESTINATION lib)
