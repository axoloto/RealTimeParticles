
project(ParticleSystemApp)

file(GLOB SRC "ParticleSystemApp.cpp" "ParticleSystemApp.hpp")
add_executable(ParticleSystemApp ${SRC})
set_target_properties(ParticleSystemApp PROPERTIES FOLDER app)

add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD)

target_include_directories(ParticleSystemApp PRIVATE ${OPENGL_INCLUDE_DIRS}
                                            ${OPENCL_INCLUDE_DIRS}
                                            "$ENV{TP_DIR}/diligentGraphics"
                                            "$ENV{TP_DIR}/glad/include" 
                                            "$ENV{TP_DIR}/SDL2/include"
                                            "$ENV{TP_DIR}/imgui")

target_link_libraries(ParticleSystemApp physics render ui imgui glad ${OPENGL_LIBRARIES} ${SDL} ${SDLmain})

install(TARGETS ParticleSystemApp RUNTIME DESTINATION bin)

# SDL2 dynamic lib is also needed for some reasons WIP
add_custom_command(TARGET ParticleSystemApp POST_BUILD COMMAND
    ${CMAKE_COMMAND} -E copy $ENV{TP_DIR}/SDL2/libx64/SDL2.dll ${CMAKE_SOURCE_DIR}/install/bin/SDL2.dll)

add_custom_command(TARGET ParticleSystemApp POST_BUILD COMMAND
    ${CMAKE_COMMAND} -E copy $ENV{TP_DIR}/SDL2/libx64/SDL2.dll ${CMAKE_SOURCE_DIR}/CMakeBuild/app/$ENV{DEV_BUILD_TYPE}/SDL2.dll)
