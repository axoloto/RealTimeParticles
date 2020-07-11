
file(GLOB SRC "*.cpp" "*.hpp")

include_directories("$ENV{TP_DIR}/glad/include")
include_directories("$ENV{TP_DIR}/SDL2/include")

add_library (imgui ${SRC})
