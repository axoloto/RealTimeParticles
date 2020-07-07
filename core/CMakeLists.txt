
file(GLOB SRC "*.cpp" "*.hpp")

add_library(boidsCore ${SRC})

install(TARGETS boidsCore RUNTIME DESTINATION bin LIBRARY DESTINATION lib)