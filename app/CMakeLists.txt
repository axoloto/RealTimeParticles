project(RealTimeParticles)

file(GLOB SRC "ParticleSystemApp.cpp" "ParticleSystemApp.hpp" "${CONAN_SRC_DIRS_IMGUI}/imgui_impl_opengl3.cpp" "${CONAN_SRC_DIRS_IMGUI}/imgui_impl_sdl.cpp")

add_executable(RealTimeParticles ${SRC})
set_target_properties(RealTimeParticles PROPERTIES FOLDER app)
set_target_properties(RealTimeParticles PROPERTIES LINK_FLAGS "/ignore:4099")

add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD)

target_include_directories(RealTimeParticles PRIVATE ${OPENGL_INCLUDE_DIRS}
                                            "$ENV{TP_DIR}/diligentGraphics" ${CONAN_SRC_DIRS_IMGUI})

conan_target_link_libraries(RealTimeParticles)
target_link_libraries(RealTimeParticles physics render ui imgui utils ${OPENGL_LIBRARIES})

install(TARGETS RealTimeParticles RUNTIME DESTINATION bin)