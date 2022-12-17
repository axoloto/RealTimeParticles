
file(GLOB SRC "*.cpp" "*.hpp")

find_package(glad REQUIRED CONFIG)

#find_package(OpenGL REQUIRED)
find_library(OpenGL_Framework OpenGL)

add_library(render ${SRC})

target_link_libraries(render PUBLIC glad::glad PRIVATE utils ${OpenGL_Framework})

install(TARGETS render DESTINATION lib)
