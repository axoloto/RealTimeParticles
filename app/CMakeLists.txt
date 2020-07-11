
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

foreach(DLL ${SDL2_DLL})
    add_custom_command(TARGET boidsApp POST_BUILD COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${SDL2_DLL} $<TARGET_FILE_DIR:boidsApp>)
endforeach()
