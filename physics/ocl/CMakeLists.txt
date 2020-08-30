file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

if(USE_OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARIES})
    add_definitions("-DOPENCL_ACTIVATED")
    message ("-- OpenCL include => ${OpenCL_INCLUDE_DIRS}")
    message ("-- OpenCL library => ${OpenCL_LIBRARIES}")
    message ("-- OpenCL version => ${OpenCL_VERSION_STRING}")
endif()

target_include_directories(ocl PRIVATE ${OPENCL_INCLUDE_DIRS})

target_link_libraries(ocl ${OpenCL_LIBRARY})

install(TARGETS ocl RUNTIME DESTINATION bin LIBRARY DESTINATION lib)