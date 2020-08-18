
file(GLOB SRC "*.cpp" "*.hpp")

add_library(physics ${SRC})

install(TARGETS physics RUNTIME DESTINATION bin LIBRARY DESTINATION lib)