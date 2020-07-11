
file(GLOB SRC "glad.c" "./include/*.hpp")

include_directories("include")

add_library (glad ${SRC})
