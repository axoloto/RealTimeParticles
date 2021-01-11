file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

install(TARGETS ocl RUNTIME DESTINATION bin LIBRARY DESTINATION lib)