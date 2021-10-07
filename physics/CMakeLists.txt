
file(GLOB SRC "*.cpp" "*.hpp")

add_library(physics ${SRC})

include_directories("ocl")
include_directories("utils")

add_subdirectory("ocl")
add_subdirectory("utils")

target_link_libraries(physics ocl utils)

install(TARGETS physics DESTINATION lib)