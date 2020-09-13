file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

target_link_libraries(ocl ${OpenCL_LIBRARY})

install(TARGETS ocl RUNTIME DESTINATION bin LIBRARY DESTINATION lib)