file(GLOB SRC "ParticleSystemApp.cpp" "ParticleSystemApp.hpp")

add_executable(RealTimeParticles ${SRC})
set_target_properties(RealTimeParticles PROPERTIES FOLDER app)

add_subdirectory(imgui_backend)

target_link_libraries(RealTimeParticles PRIVATE physics render ui imgui_backend utils)

if(APPLE)
    find_library(OpenGL_Framework OpenGL)
    target_link_libraries(RealTimeParticles PRIVATE ${OpenGL_Framework})
elseif()
    find_package(OpenGL REQUIRED)
    target_link_libraries(RealTimeParticles PRIVATE OpenGL)
endif()

if(WIN32)
    set_target_properties(RealTimeParticles PROPERTIES LINK_FLAGS "/ignore:4099")
endif()

install(TARGETS RealTimeParticles RUNTIME DESTINATION bin)