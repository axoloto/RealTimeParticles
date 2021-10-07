file(GLOB SRC "*.cpp" "*.hpp")

add_library(ocl ${SRC})

install(TARGETS ocl DESTINATION lib)
#Adding kernels to use them with packaged installer
install(DIRECTORY kernels DESTINATION bin)