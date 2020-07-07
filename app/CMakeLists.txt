
#find_package(GLEW required)
#find_package(glfw3 required)
#find_package(OpenGL required)

file(GLOB SRC "BoidsApp.cpp")
add_executable(boidsApp ${SRC})

target_link_libraries(boidsApp boidsCore)

install(TARGETS boidsApp RUNTIME DESTINATION bin)