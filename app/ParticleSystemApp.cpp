
#include "ParticleSystemApp.hpp"
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>

#include <glad/glad.h>
#include <sdl2/SDL.h>
#include <spdlog/spdlog.h>

#include "Boids.hpp"

#include "RadixSortCommon.hpp"

#if __APPLE__
constexpr auto GLSL_VERSION = "#version 150";
#else
constexpr auto GLSL_VERSION
    = "#version 130";
#endif

bool ParticleSystemApp::initWindow()
{
  // Setup SDL
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
  {
    spdlog::error("Error: {}", SDL_GetError());
    return false;
  }

  // GL 3.0 + GLSL 130
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

  // Create window with graphics context
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  m_window = SDL_CreateWindow(m_nameApp.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_windowSize.x, m_windowSize.y, window_flags);
  m_OGLContext = SDL_GL_CreateContext(m_window);
  SDL_GL_MakeCurrent(m_window, m_OGLContext);
  SDL_GL_SetSwapInterval(1); // Enable vsync

  // Initialize OpenGL loader
  bool err = gladLoadGL() == 0;

  if (err)
  {
    spdlog::error("Failed to initialize OpenGL loader!");
    return false;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGui::StyleColorsDark();

  ImGui_ImplSDL2_InitForOpenGL(m_window, m_OGLContext);
  ImGui_ImplOpenGL3_Init(GLSL_VERSION);

  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  //
  //
  //

  // create instance for device
  SimpleRadixSort64bit rs { getDevice() };

  std::vector<SimpleRadixSort64bit::KeyType> v(1 << 9);
  std::cout << "generating " << v.size() << " keys...." << std::endl;

  auto rng = makeRng(rs.maxValue());
  std::generate(v.begin(), v.end(), rng);

  // take a copy of v for permutation check
  std::vector<SimpleRadixSort64bit::KeyType> vCopy = v;

  // try
  {
    // sort method is STL-style
    rs.sort(v.begin(), v.end());
  }
  // catch (const cl::Error& e)
  {
    //   std::cerr << "CL Error: " << e.err() << " " << e.what() << std::endl;
    //   return 1;
  }

  std::cout << "Sorted: " << std::boolalpha << std::is_sorted(v.begin(), v.end()) << std::endl;
  std::cout << "Permutation correct: " << std::boolalpha << checkPermutation(v, vCopy, rs.permutation()) << std::endl;

  const auto pi = rs.profilingInfo();
  std::cout << "Profiling: " << std::endl;

  std::cout << "total kernel execution time: " << pi.total() * 1e-6 << "ms" << std::endl;

  //
  //
  //

  return true;
}

bool ParticleSystemApp::closeWindow()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(m_OGLContext);
  SDL_DestroyWindow(m_window);
  SDL_Quit();

  return true;
}

void ParticleSystemApp::checkMouseState()
{
  ImGuiIO& io = ImGui::GetIO();
  (void)io;

  if (io.WantCaptureMouse)
    return;

  Math::int2 currentMousePos;
  auto mouseState = SDL_GetMouseState(&currentMousePos.x, &currentMousePos.y);
  Math::int2 delta = currentMousePos - m_mousePrevPos;
  Math::float2 fDelta((float)delta.x, (float)delta.y);

  if (mouseState & SDL_BUTTON(1))
  {
    m_graphicsEngine->checkMouseEvents(Render::UserAction::ROTATION, fDelta);
    m_mousePrevPos = currentMousePos;
  }
  else if (mouseState & SDL_BUTTON(3))
  {
    m_graphicsEngine->checkMouseEvents(Render::UserAction::TRANSLATION, fDelta);
    m_mousePrevPos = currentMousePos;
  }
}

bool ParticleSystemApp::checkSDLStatus()
{
  bool stopRendering = false;

  SDL_Event event;
  while (SDL_PollEvent(&event))
  {
    ImGui_ImplSDL2_ProcessEvent(&event);
    switch (event.type)
    {
    case SDL_QUIT:
      stopRendering = true;
      break;
    case SDL_WINDOWEVENT:
      if (event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(m_window))
      {
        stopRendering = true;
      }
      if (event.window.event == SDL_WINDOWEVENT_RESIZED)
      {
        m_windowSize = Math::int2(event.window.data1, event.window.data2);
        m_graphicsEngine->setWindowSize(m_windowSize);
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
    case SDL_MOUSEWHEEL:
      if (event.wheel.y > 0)
      {
        m_graphicsEngine->checkMouseEvents(Render::UserAction::ZOOM, Math::float2(-0.4f, 0.f));
      }
      else if (event.wheel.y < 0)
      {
        m_graphicsEngine->checkMouseEvents(Render::UserAction::ZOOM, Math::float2(0.4f, 0.f));
      }
      break;
    }
  }
  return stopRendering;
}

ParticleSystemApp::ParticleSystemApp()
    : m_nameApp("Particle System Sandbox")
    , m_mousePrevPos(0, 0)
    , m_backGroundColor(0.0f, 0.0f, 0.0f, 1.00f)
    , m_buttonRightActivated(false)
    , m_buttonLeftActivated(false)
    , m_windowSize(1280, 720)
    , m_init(false)
    , m_boxSize(500)
    , m_gridRes(10)
    , m_numEntities(Core::NUM_MAX_ENTITIES)
{
  initWindow();

  m_graphicsEngine = std::make_unique<Render::OGLRender>(m_boxSize, m_gridRes, m_numEntities, Core::NUM_MAX_ENTITIES, (float)m_windowSize.x / m_windowSize.y);

  if (!m_graphicsEngine)
    return;

  m_physicsEngine = std::make_unique<Core::Boids>(m_numEntities, m_gridRes,
      (unsigned int)m_graphicsEngine->pointCloudCoordVBO(),
      (unsigned int)m_graphicsEngine->pointCloudColorVBO(),
      (unsigned int)m_graphicsEngine->gridDetectorVBO());

  if (!m_physicsEngine)
    return;

  m_physicsWidget = std::make_unique<UI::BoidsWidget>(*m_physicsEngine);

  if (!m_physicsWidget)
    return;

  m_init = true;
}

void ParticleSystemApp::run()
{
  ImGuiIO& io = ImGui::GetIO();
  (void)io;

  bool stopRendering = false;
  while (!stopRendering)
  {
    stopRendering = checkSDLStatus();

    checkMouseState();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(m_window);
    ImGui::NewFrame();

    if (!m_physicsEngine->isInit())
    {
      stopRendering = popUpErrorMessage("The application needs OpenCL 1.2 or more recent to run.");
    }

    displayMainWidget();

    m_physicsWidget->display();

    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    glClearColor(m_backGroundColor.x, m_backGroundColor.y, m_backGroundColor.z, m_backGroundColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_physicsEngine->update();

    m_graphicsEngine->draw();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(m_window);
  }

  closeWindow();
}

void ParticleSystemApp::displayMainWidget()
{
  ImGui::Begin("Main Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  bool isOnPaused = m_physicsEngine->onPause();
  std::string pauseRun = isOnPaused ? "  Start  " : "  Pause  ";
  if (ImGui::Button(pauseRun.c_str()))
  {
    m_physicsEngine->pause(!isOnPaused);
  }

  ImGui::SameLine();

  if (ImGui::Button("  Reset  "))
  {
    m_physicsEngine->reset();
  }

  if (ImGui::SliderInt("Particles", &m_numEntities, 1, Core::NUM_MAX_ENTITIES))
  {
    m_physicsEngine->setNumEntities(m_numEntities);
    m_physicsEngine->reset();
    m_graphicsEngine->setNumDisplayedEntities(m_numEntities);
  }

  bool isSystemDim2D = (m_physicsEngine->dimension() == Core::Dimension::dim2D);
  if (ImGui::Checkbox("2D", &isSystemDim2D))
  {
    m_physicsEngine->setDimension(isSystemDim2D ? Core::Dimension::dim2D : Core::Dimension::dim3D);
  }

  ImGui::SameLine();

  bool isSystemDim3D = (m_physicsEngine->dimension() == Core::Dimension::dim3D);
  if (ImGui::Checkbox("3D", &isSystemDim3D))
  {
    m_physicsEngine->setDimension(isSystemDim3D ? Core::Dimension::dim3D : Core::Dimension::dim2D);
  }

  ImGui::SameLine();

  bool isBoxVisible = m_graphicsEngine->isBoxVisible();
  if (ImGui::Checkbox("Box", &isBoxVisible))
  {
    m_graphicsEngine->setBoxVisibility(isBoxVisible);
  }

  ImGui::SameLine();

  bool isGridVisible = m_graphicsEngine->isGridVisible();
  if (ImGui::Checkbox("Grid", &isGridVisible))
  {
    m_graphicsEngine->setGridVisibility(isGridVisible);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  float velocity = m_physicsEngine->velocity();
  if (ImGui::SliderFloat("Speed", &velocity, 0.01f, 10.0f))
  {
    m_physicsEngine->setVelocity(velocity);
  }

  const auto cameraPos = m_graphicsEngine->cameraPos();
  const auto targetPos = m_graphicsEngine->targetPos();

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::Text(" Camera (%.1f, %.1f, %.1f)", cameraPos.x, cameraPos.y, cameraPos.z);
  ImGui::Text(" Target (%.1f, %.1f, %.1f)", targetPos.x, targetPos.y, targetPos.z);
  ImGui::Text(" Dist. camera target : %.1f", Math::length(cameraPos - targetPos));
  ImGui::Spacing();
  if (ImGui::Button(" Reset Camera "))
  {
    m_graphicsEngine->resetCamera();
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::Text(" %.3f ms/frame (%.1f FPS) ", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
  ImGui::End();
}

bool ParticleSystemApp::popUpErrorMessage(std::string errorMessage)
{
  bool closePopUp = false;

  bool open = true;
  ImGui::OpenPopup("Error");
  if (ImGui::BeginPopupModal("Error", &open))
  {
    ImGui::Text(errorMessage.c_str());
    if (ImGui::Button((std::string("Close ") + m_nameApp).c_str()))
    {
      ImGui::CloseCurrentPopup();
      closePopUp = true;
    }
    ImGui::EndPopup();
  }

  return closePopUp;
}

auto initializeLogger()
{
  spdlog::set_level(spdlog::level::debug);
}

int main(int, char**)
{
  initializeLogger();

  ParticleSystemApp app;

  if (app.isInit())
  {
    app.run();
  }

  return 0;
}