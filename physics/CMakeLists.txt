
file(GLOB SRC "*.cpp" "*.hpp")

add_library(physics ${SRC})

include_directories("ocl")
include_directories("utils")

add_subdirectory("ocl")
add_subdirectory("utils")

# Not the physics/utils folder but the generic static utils lib
target_link_libraries(physics PRIVATE utils)

target_include_directories(physics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS physics DESTINATION lib)