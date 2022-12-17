file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

target_link_libraries(ocl PRIVATE utils OpenCL)

if(UNIX AND NOT APPLE)
    target_link_libraries(ocl PRIVATE OpenGL::GLX)
elseif(APPLE)
    find_library(OclFramework OpenCL) # Should be in System/Library/Frameworks/OpenCL.framework
    target_link_libraries(ocl PRIVATE ${OclFramework})
endif()

install(TARGETS ocl DESTINATION lib)

#Adding kernels to use them with packaged installer
install(DIRECTORY kernels DESTINATION bin)