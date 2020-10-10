
project(ParticleSystemApp)

file(GLOB SRC "ParticleSystemApp.cpp" "ParticleSystemApp.hpp")

add_executable(ParticleSystemApp ${SRC})
set_target_properties(ParticleSystemApp PROPERTIES FOLDER app)
set_target_properties(ParticleSystemApp PROPERTIES LINK_FLAGS "/ignore:4099")

add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD)

target_include_directories(ParticleSystemApp PRIVATE ${OPENGL_INCLUDE_DIRS}
                                            "$ENV{TP_DIR}/diligentGraphics"
                                            "$ENV{TP_DIR}/imgui")


conan_target_link_libraries(ParticleSystemApp)
target_link_libraries(ParticleSystemApp physics render ui imgui ${OPENGL_LIBRARIES})

install(TARGETS ParticleSystemApp RUNTIME DESTINATION bin)