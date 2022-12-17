
file(GLOB SRC "*.cpp" "*.hpp")

add_library(physics ${SRC})

include_directories("ocl")
include_directories("utils")

add_subdirectory("ocl")
add_subdirectory("utils")

target_link_libraries(physics PRIVATE ocl utils)

target_include_directories(physics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS physics DESTINATION lib)