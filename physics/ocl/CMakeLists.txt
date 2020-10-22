file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

#include_directories(${OpenCL_INCLUDE_DIRS})

#target_link_libraries(ocl ${OpenCL_LIBRARY})

install(TARGETS ocl RUNTIME DESTINATION bin LIBRARY DESTINATION lib)