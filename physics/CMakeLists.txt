
file(GLOB SRC "*.cpp" "*.hpp")

#add_library(physics ${SRC})

if(USE_OPENCL)
  include_directories("ocl")
  add_subdirectory("ocl")
#	target_link_libraries(physics ocl)
endif()

#install(TARGETS physics RUNTIME DESTINATION bin LIBRARY DESTINATION lib)