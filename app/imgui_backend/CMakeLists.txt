file(GLOB SRC "*.cpp" "*.hpp")

add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD)

find_package(imgui REQUIRED CONFIG)
find_package(sdl2 REQUIRED CONFIG)

add_library(imgui_backend ${SRC})

target_include_directories(imgui_backend INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(imgui_backend PUBLIC imgui::imgui SDL2::SDL2)

install(TARGETS imgui_backend DESTINATION lib)