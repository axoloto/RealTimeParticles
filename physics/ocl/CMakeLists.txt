file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

target_compile_definitions(ocl PUBLIC CL_HPP_ENABLE_EXCEPTIONS)
target_compile_definitions(ocl PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=120)
target_compile_definitions(ocl PUBLIC CL_HPP_TARGET_OPENCL_VERSION=120)

target_link_libraries(ocl PRIVATE utils)

if(WIN32)
    target_link_libraries(ocl PRIVATE OpenCL)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(ocl PRIVATE OpenCL OpenGL::GLX)
elseif(APPLE)
    find_library(OclFramework OpenCL) # Should be in System/Library/Frameworks/OpenCL.framework
    target_link_libraries(ocl PRIVATE ${OclFramework})
endif()

install(TARGETS ocl DESTINATION lib)

#Adding kernels to use them with packaged installer
install(DIRECTORY kernels DESTINATION bin)