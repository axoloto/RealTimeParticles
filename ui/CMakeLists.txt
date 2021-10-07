
file(GLOB SRC "*.cpp" "*.hpp")

add_library(ui ${SRC})

install(TARGETS ui DESTINATION lib)