
file(GLOB SRC "*.cpp" "*.hpp")

find_package(imgui REQUIRED CONFIG)

add_library(ui ${SRC})

target_link_libraries(ui PUBLIC imgui::imgui PRIVATE render physics utils)

target_include_directories(ui INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ui DESTINATION lib)