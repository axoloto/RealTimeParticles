file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

target_compile_definitions(ocl PUBLIC CL_HPP_ENABLE_EXCEPTIONS)
target_compile_definitions(ocl PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=120)
target_compile_definitions(ocl PUBLIC CL_HPP_TARGET_OPENCL_VERSION=120)

target_link_libraries(ocl PRIVATE utils)

target_include_directories(ocl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT APPLE)
    find_package(OpenCLHeaders REQUIRED CONFIG)
    find_package(OpenCLICDLoader REQUIRED CONFIG)
    target_link_libraries(ocl PRIVATE OpenCL::Headers OpenCL::OpenCL)
elseif(APPLE)
    find_library(OclFramework OpenCL) # Should be in System/Library/Frameworks/OpenCL.framework
    target_link_libraries(ocl PRIVATE ${OclFramework})
endif()

install(TARGETS ocl DESTINATION lib)

#Adding kernels to use them with packaged installer
install(DIRECTORY kernels DESTINATION bin)