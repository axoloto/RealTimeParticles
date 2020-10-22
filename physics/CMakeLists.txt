
file(GLOB SRC "*.cpp" "*.hpp")

add_library(physics ${SRC})

include_directories("ocl")
add_subdirectory("ocl")
target_link_libraries(physics ocl)

install(TARGETS physics RUNTIME DESTINATION bin LIBRARY DESTINATION lib)