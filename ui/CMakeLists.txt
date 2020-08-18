
file(GLOB SRC "*.cpp" "*.hpp")

add_library(ui ${SRC})

install(TARGETS ui RUNTIME DESTINATION bin LIBRARY DESTINATION lib)