file(GLOB SRC "*.cpp" "*.hpp")
target_sources(physics PRIVATE ${SRC})

target_compile_definitions(physics PUBLIC CL_HPP_ENABLE_EXCEPTIONS)
target_compile_definitions(physics PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=120)
target_compile_definitions(physics PUBLIC CL_HPP_TARGET_OPENCL_VERSION=120)

if(APPLE)
    find_library(OclFramework OpenCL) # Should be in System/Library/Frameworks/OpenCL.framework
    target_link_libraries(physics PRIVATE ${OclFramework})
else()
    find_package(OpenCLHeaders REQUIRED CONFIG)
    find_package(OpenCLICDLoader REQUIRED CONFIG)
    target_link_libraries(physics PRIVATE OpenCL::Headers OpenCL::OpenCL)
endif()

if(MSVC)
    # Should be somewhere on local system, needed for CL-GL interop
    find_package(OpenGL REQUIRED)
    target_include_directories(physics PRIVATE ${OPENGL_INCLUDE_DIRS})
    target_link_libraries(physics PRIVATE ${OPENGL_LIBRARIES})
endif()

#Adding kernels to use them with packaged installer
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/kernels" DESTINATION bin)