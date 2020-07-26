#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl.h"
#include "imgui/imgui_impl_opengl3.h"
#include <stdio.h>
#include <SDL.h>
#include <glad/glad.h>
#include <iostream>

#include "BoidsGenerator.hpp"
#include "BoidsApp.hpp"

bool BoidsApp::initOGL()
{
    // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return false;
    }

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    m_window = SDL_CreateWindow("Boids Visualizer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
    m_OGLContext = SDL_GL_CreateContext(m_window);
    SDL_GL_MakeCurrent(m_window, m_OGLContext);
    SDL_GL_SetSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
    bool err = gladLoadGL() == 0;

    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForOpenGL(m_window, m_OGLContext);
    ImGui_ImplOpenGL3_Init(glsl_version);

    return true;
}

bool BoidsApp::closeOGL()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(m_OGLContext);
    SDL_DestroyWindow(m_window);
    SDL_Quit();

    return true;
}

BoidsApp::BoidsApp()
{
    initOGL();

    m_OGLRender = std::make_unique<Render::OGLRender>();

    //auto boidsGenerator = Core::BoidsGenerator(1000);

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Load Fonts
   // ImFont* font = io.Fonts->AddFontFromFileTTF("assets/verdana.ttf", 18.0f, NULL, NULL);
   //IM_ASSERT(font != NULL);

    m_backGroundColor = ImVec4(0.85f, 0.55f, 0.60f, 1.00f);
}

void BoidsApp::run()
{
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    bool stopRendering = false;
    while (!stopRendering)
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            switch(event.type)
            { 
            case SDL_QUIT :
                stopRendering = true;
            case SDL_WINDOWEVENT :
                if(event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(m_window))
                    stopRendering = true;
            case SDL_MOUSEBUTTONDOWN :
                if(event.button.button == SDL_BUTTON_LEFT)
                    m_buttonLeftActivated = true;
                else if(event.button.button == SDL_BUTTON_RIGHT)
                    m_buttonRightActivated = true;
            case SDL_MOUSEBUTTONUP :
                if(event.button.button == SDL_BUTTON_LEFT)
                    m_buttonLeftActivated = false;
                else if(event.button.button == SDL_BUTTON_RIGHT)
                    m_buttonRightActivated = false;
            case SDL_MOUSEMOTION :
                if(m_buttonLeftActivated)
                {}
            }
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(m_window);
        ImGui::NewFrame();

        ImGui::Begin("Boids Simulator");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(m_backGroundColor.x, m_backGroundColor.y, m_backGroundColor.z, m_backGroundColor.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_OGLRender->draw();

        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(m_window);
    }

    closeOGL();
}

int main(int, char**)
{
    BoidsApp app;

    app.run();

    return 0;
}