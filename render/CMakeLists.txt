
file(GLOB SRC "*.cpp" "*.hpp")

add_library (render ${SRC})

target_include_directories(render PRIVATE ${OPENGL_INCLUDE_DIRS}
                                            "$ENV{TP_DIR}/glad/include"
                                            "$ENV{TP_DIR}/diligentGraphics"
                                            "$ENV{TP_DIR}/imgui/include")
