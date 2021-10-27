file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

install(TARGETS ocl DESTINATION lib)
if(UNIX)
    target_link_libraries(ocl OpenCL OpenGL::GLX)
endif()
#Adding kernels to use them with packaged installer
install(DIRECTORY kernels DESTINATION bin)