
file(GLOB SRC "*.cpp" "*.hpp")

find_package(imgui REQUIRED CONFIG)

add_library(ui ${SRC})

target_link_libraries(ui PUBLIC imgui::imgui PRIVATE render utils)

install(TARGETS ui DESTINATION lib)