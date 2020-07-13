
project(boidsApp)

file(GLOB SRC "BoidsApp.cpp")
add_executable(boidsApp ${SRC})
set_target_properties(boidsApp PROPERTIES FOLDER app)

add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD)

target_include_directories(boidsApp PRIVATE ${OPENGL_INCLUDE_DIRS}
                                            "$ENV{TP_DIR}/glad/include" 
                                            "$ENV{TP_DIR}/SDL2/include"
                                            "$ENV{TP_DIR}/imgui")

target_link_libraries(boidsApp boidsCore imgui glad ${OPENGL_LIBRARIES} ${SDL} ${SDLmain})
#target_link_libraries(boidsApp boidsCore imgui glad ${OPENGL_LIBRARIES})

install(TARGETS boidsApp RUNTIME DESTINATION bin)

# SDL2 dynamic lib is also needed for some reasons WIP
add_custom_command(TARGET boidsApp POST_BUILD COMMAND
    ${CMAKE_COMMAND} -E copy $ENV{TP_DIR}/SDL2/libx64/SDL2.dll ${CMAKE_SOURCE_DIR}/install/bin/SDL2.dll)
