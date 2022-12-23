
file(GLOB SRC "*.cpp" "*.hpp")

find_package(glad REQUIRED CONFIG)

add_library(render ${SRC})

target_link_libraries(render PUBLIC glad::glad PRIVATE utils)

if(APPLE)
    find_library(OpenGL_Framework OpenGL)
    target_link_libraries(render PRIVATE ${OpenGL_Framework})
endif()

target_include_directories(render INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS render DESTINATION lib)
