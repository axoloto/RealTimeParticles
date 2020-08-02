#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl.h"
#include "imgui/imgui_impl_opengl3.h"
#include <stdio.h>
#include <SDL.h>
#include <glad/glad.h>
#include <iostream>

#include "BoidsGenerator.hpp"
#include "BoidsApp.hpp"

bool BoidsApp::initWindow()
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
    m_window = SDL_CreateWindow("Particle System Sandbox", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_windowSize.x, m_windowSize.y, window_flags);
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

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Load Fonts
    // ImFont* font = io.Fonts->AddFontFromFileTTF("assets/verdana.ttf", 18.0f, NULL, NULL);
    //IM_ASSERT(font != NULL);

    return true;
}

bool BoidsApp::closeWindow()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(m_OGLContext);
    SDL_DestroyWindow(m_window);
    SDL_Quit();

    return true;
}

void BoidsApp::checkMouseState()
{
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    if(io.WantCaptureMouse) return;

    Math::int2 currentMousePos;
    auto mouseState = SDL_GetMouseState(&currentMousePos.x, &currentMousePos.y);
    Math::int2 delta = currentMousePos - m_mousePrevPos;
    Math::float2 fDelta((float)delta.x, (float)delta.y);

    if(mouseState & SDL_BUTTON(1))
    {
        m_OGLRender->checkMouseEvents(Render::UserAction::ROTATION, fDelta);
        m_mousePrevPos = currentMousePos;
    }
    else if(mouseState & SDL_BUTTON(3))
    {
        m_OGLRender->checkMouseEvents(Render::UserAction::TRANSLATION, fDelta);
        m_mousePrevPos = currentMousePos;
    }
}

bool BoidsApp::checkSDLStatus()
{
    bool stopRendering = false;

    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL2_ProcessEvent(&event);
        switch(event.type)
        { 
            case SDL_QUIT :
                stopRendering = true;
                break;
            case SDL_WINDOWEVENT :
                if(event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(m_window))
                {
                    stopRendering = true;
                }
                if (event.window.event == SDL_WINDOWEVENT_RESIZED)
                {
                    m_windowSize = Math::int2(event.window.data1, event.window.data2);
                    m_OGLRender->setWindowSize(m_windowSize);
                }
                break;
            case SDL_MOUSEBUTTONDOWN:
                {
                    if (event.button.button == SDL_BUTTON_LEFT || event.button.button == SDL_BUTTON_RIGHT)
                    {      
                        Math::int2 currentMousePos;
                        SDL_GetMouseState(&currentMousePos.x, &currentMousePos.y);
                        m_mousePrevPos = currentMousePos;
                    }
                }
                break;
            case SDL_MOUSEWHEEL :
                if(event.wheel.y > 0) 
                {
                    m_OGLRender->checkMouseEvents(Render::UserAction::ZOOM, Math::float2(-0.2f, 0.f));
                }
                else if(event.wheel.y < 0) 
                {
                    m_OGLRender->checkMouseEvents(Render::UserAction::ZOOM, Math::float2(0.2f, 0.f));
                }
                break;
        }
    }
    return stopRendering;
}

BoidsApp::BoidsApp() : m_mousePrevPos(0, 0), m_backGroundColor(0.0f, 0.0f, 0.0f, 1.00f), m_buttonRightActivated(false), m_buttonLeftActivated(false), m_windowSize(1280, 720), m_init(false)
{
    initWindow();

    int halfBoxSize = 200;
    m_boidsGenerator = std::make_unique<Core::BoidsGenerator>(halfBoxSize);

    if(!m_boidsGenerator) return;

    m_OGLRender = std::make_unique<Render::OGLRender>(halfBoxSize, 3000, (float) m_windowSize.x / m_windowSize.y);

    if(!m_OGLRender) return;

    m_OGLRender->setPointCloudBuffer(m_boidsGenerator->getVerticesBufferStart(), m_boidsGenerator->getVerticesBufferSize());

    m_init = true;
}

void BoidsApp::run()
{
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    bool stopRendering = false;
    while (!stopRendering)
    {
        stopRendering = checkSDLStatus();

        checkMouseState();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(m_window);
        ImGui::NewFrame();

        if(!m_OGLRender)
        {
            stopRendering = true;
            return;
        }

        const auto cameraPos = m_OGLRender->cameraPos();
        const auto targetPos = m_OGLRender->targetPos();

        ImGui::Begin("Main Widget");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text(" Camera position : %.1f x, %.1f y, %.1f z", cameraPos.x, cameraPos.y, cameraPos.z);
        ImGui::Text(" Target position : %.1f x, %.1f y, %.1f z", targetPos.x, targetPos.y, targetPos.z);
        ImGui::Text(" Distance camera target : %.1f", Math::length(cameraPos - targetPos));
        ImGui::Spacing();
        if(ImGui::Button(" Reset Camera "))
        { 
            m_OGLRender->resetCamera();
        }
        ImGui::End();

        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(m_backGroundColor.x, m_backGroundColor.y, m_backGroundColor.z, m_backGroundColor.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_boidsGenerator->updateBoids();

        m_OGLRender->draw();

        ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(m_window);
    }

    closeWindow();
}

int main(int, char**)
{
    BoidsApp app;

    if(app.isInit())
    {
        app.run();
    }

    return 0;
}