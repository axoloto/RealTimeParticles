
#include "ParticleSystemApp.hpp"

#include "Logging.hpp"
#include "Model.hpp"
#include "Parameters.hpp"
#include "Utils.hpp"

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl.h>

#include <SDL2/SDL.h>
#include <glad/glad.h>

#if __APPLE__
constexpr auto GLSL_VERSION = "#version 150";
#else
constexpr auto GLSL_VERSION = "#version 130";
#endif

namespace App
{
bool ParticleSystemApp::initWindow()
{
  // Setup SDL
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
  {
    LOG_ERROR("Error: {}", SDL_GetError());
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
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_SHOWN);
  m_window = SDL_CreateWindow(m_nameApp.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_windowSize.x, m_windowSize.y, window_flags);
  m_OGLContext = SDL_GL_CreateContext(m_window);
  SDL_GL_MakeCurrent(m_window, m_OGLContext);
  SDL_GL_SetSwapInterval(1); // Enable vsync

  // Attempt to fit app to full working space of current window
  int displayIndex = SDL_GetWindowDisplayIndex(m_window);
  if (displayIndex >= 0)
  {
    int topBorderSize = 0;
    SDL_GetWindowBordersSize(m_window, &topBorderSize, NULL, NULL, NULL);

    SDL_Rect usableBounds;
    if (SDL_GetDisplayUsableBounds(displayIndex, &usableBounds) == 0)
    {
      SDL_SetWindowPosition(m_window, usableBounds.x, usableBounds.y + topBorderSize);
      SDL_SetWindowSize(m_window, usableBounds.w, usableBounds.h - topBorderSize);
      m_windowSize = Math::int2(usableBounds.w, usableBounds.h - topBorderSize);
    }
  }

  // Initialize OpenGL loader
  bool err = gladLoadGL() == 0;

  if (err)
  {
    LOG_ERROR("Failed to initialize OpenGL loader!");
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
    {
      stopRendering = true;
      break;
    }
    case SDL_WINDOWEVENT:
    {
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
    }
    case SDL_MOUSEBUTTONDOWN:
    {
      if (event.button.button == SDL_BUTTON_LEFT || event.button.button == SDL_BUTTON_RIGHT)
      {
        Math::int2 currentMousePos;
        SDL_GetMouseState(&currentMousePos.x, &currentMousePos.y);
        m_mousePrevPos = currentMousePos;
      }
      break;
    }
    case SDL_MOUSEWHEEL:
    {
      if (event.wheel.y > 0)
      {
        m_graphicsEngine->checkMouseEvents(Render::UserAction::ZOOM, Math::float2(-1.2f, 0.f));
      }
      else if (event.wheel.y < 0)
      {
        m_graphicsEngine->checkMouseEvents(Render::UserAction::ZOOM, Math::float2(1.2f, 0.f));
      }
      break;
    }
    case SDL_KEYDOWN:
    {
      bool isPaused = m_physicsEngine->onPause();
      m_physicsEngine->pause(!isPaused);
      break;
    }
    }
  }
  return stopRendering;
}

ParticleSystemApp::ParticleSystemApp()
    : m_nameApp("RealTimeParticles " + Utils::GetVersions())
    , m_mousePrevPos(0, 0)
    , m_backGroundColor(0.0f, 0.0f, 0.0f, 1.00f)
    , m_buttonRightActivated(false)
    , m_buttonLeftActivated(false)
    , m_windowSize(1280, 720)
    , m_modelType(Physics::ModelType::FLUIDS)
    , m_targetFps(60)
    , m_currFps(60.0f)
    , m_init(false)
{
  LOG_INFO("Starting RealTimeParticles");

  if (!initWindow())
  {
    LOG_ERROR("Failed to initialize application window");
    return;
  }

  if (!initGraphicsEngine())
  {
    LOG_ERROR("Failed to initialize graphics engine");
    return;
  }

  if (!initGraphicsWidget())
  {
    LOG_ERROR("Failed to initialize graphics widget");
    return;
  }

  if (!initPhysicsEngine())
  {
    LOG_ERROR("Failed to initialize physics engine");
    return;
  }

  if (!initPhysicsWidget())
  {
    LOG_ERROR("Failed to initialize physics widget");
    return;
  }

  LOG_INFO("RealTimeParticles initialization successful");

  m_init = true;
}

ParticleSystemApp::~ParticleSystemApp()
{
  LOG_INFO("Quitting RealTimeParticles");
}

bool ParticleSystemApp::initGraphicsEngine()
{
  Render::EngineParams params;
  params.maxNbParticles = Utils::ALL_NB_PARTICLES.crbegin()->first;
  params.boxSize = Geometry::BOX_SIZE_3D;
  params.gridRes = Geometry::GRID_RES_3D;
  params.aspectRatio = (float)m_windowSize.x / m_windowSize.y;
  params.dimension = (m_graphicsEngine.get() != nullptr) ? m_graphicsEngine->dimension() : Geometry::Dimension::dim2D;

  if (m_modelType == Physics::ModelType::CLOUDS)
  {
    params.boxSize.y *= 2;
    params.gridRes.y *= 2;
  }
  else if (m_modelType == Physics::ModelType::BOIDS)
  {
    params.pointSize = 2;
  }

  m_graphicsEngine = std::make_unique<Render::Engine>(params);

  return (m_graphicsEngine.get() != nullptr);
}

bool ParticleSystemApp::initGraphicsWidget()
{
  m_graphicsWidget = std::make_unique<UI::GraphicsWidget>(m_graphicsEngine.get());

  return (m_graphicsWidget.get() != nullptr);
}

bool ParticleSystemApp::initPhysicsEngine()
{
  Physics::ModelParams params;
  params.maxNbParticles = Utils::ALL_NB_PARTICLES.crbegin()->first;
  params.boxSize = Geometry::BOX_SIZE_3D;
  params.gridRes = Geometry::GRID_RES_3D;
  params.velocity = 1.0f;
  params.particlePosVBO = (unsigned int)m_graphicsEngine->pointCloudCoordVBO();
  params.particleColVBO = (unsigned int)m_graphicsEngine->pointCloudColorVBO();
  params.cameraVBO = (unsigned int)m_graphicsEngine->cameraCoordVBO();
  params.gridVBO = (unsigned int)m_graphicsEngine->gridDetectorVBO();
  params.dimension = m_graphicsEngine->dimension();

  switch (m_modelType)
  {
  case Physics::BOIDS:
  {
    params.pCase = Utils::PhysicsCase::BOIDS_SMALL;
    break;
  }
  case Physics::FLUIDS:
  {
    params.pCase = Utils::PhysicsCase::FLUIDS_DAM;
    break;
  }
  case Physics::CLOUDS:
  {
    params.pCase = Utils::PhysicsCase::CLOUDS_CUMULUS;
    params.boxSize.y *= 2;
    params.gridRes.y *= 2;
    break;
  }
  }

  if (m_physicsEngine)
  {
    LOG_DEBUG("Physics engine already existing, resetting it");
    m_physicsEngine.reset();
  }

  m_physicsEngine = Physics::CreateModel(m_modelType, params);

  return (m_physicsEngine.get() != nullptr);
}

bool ParticleSystemApp::initPhysicsWidget()
{
  m_physicsWidget = std::make_unique<UI::PhysicsWidget>(m_physicsEngine);

  return (m_physicsWidget.get() != nullptr);
}

void ParticleSystemApp::run()
{
  auto start = std::chrono::steady_clock::now();

  bool stopRendering = false;
  while (!stopRendering)
  {
    stopRendering = checkSDLStatus();

    checkMouseState();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(m_window);
    ImGui::NewFrame();

    static bool noted = false;
    if (!noted && m_physicsEngine->isUsingIGPU())
    {
      noted = popUpMessage("Warning", "The application is currently running on your integrated GPU. It will perform better on your dedicated GPU (NVIDIA/AMD).");
    }

    if (!m_physicsEngine->isInit())
    {
      stopRendering = popUpMessage("Error", "The application needs OpenCL 1.2 or more recent to run.");
    }

    displayMainWidget();

    m_graphicsWidget->display();
    m_physicsWidget->display();

    ImGuiIO& io = ImGui::GetIO();

#ifdef __APPLE__
    // On Apple, window size is reported in low DPI, even when running in high DPI mode
    glViewport(0, 0, (int)io.DisplaySize.x * io.DisplayFramebufferScale.x, (int)io.DisplaySize.y * io.DisplayFramebufferScale.y);
#else
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
#endif

    glClearColor(m_backGroundColor.x, m_backGroundColor.y, m_backGroundColor.z, m_backGroundColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Slowing down physics engine if target fps is lower than current fps
    auto now = std::chrono::steady_clock::now();
    auto timeSpent = now - start;
    if (timeSpent > std::chrono::milliseconds(1000 / m_targetFps))
    {
      m_currFps = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(timeSpent).count();

      m_physicsEngine->update();

      m_graphicsEngine->setNbParticles((int)m_physicsEngine->nbParticles());
      m_graphicsEngine->setTargetVisibility(m_physicsEngine->isTargetVisible());
      m_graphicsEngine->setTargetPos(m_physicsEngine->targetPos());

      start = now;
    }

    m_graphicsEngine->draw();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(m_window);
  }

  closeWindow();
}

void ParticleSystemApp::displayMainWidget()
{
  // First default pos
  ImGui::SetNextWindowPos(ImVec2(15, 12), ImGuiCond_FirstUseEver);

  ImGui::Begin("Main Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  if (!isInit())
    return;

  if (!selectPhysicalModel())
    return;

  selectPhysicalCase();

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

  bool isSystemDim2D = (m_physicsEngine->dimension() == Geometry::Dimension::dim2D);
  if (ImGui::Checkbox("2D", &isSystemDim2D))
  {
    m_physicsEngine->setDimension(isSystemDim2D ? Geometry::Dimension::dim2D : Geometry::Dimension::dim3D);
    m_graphicsEngine->setDimension(isSystemDim2D ? Geometry::Dimension::dim2D : Geometry::Dimension::dim3D);
  }

  ImGui::SameLine();

  bool isSystemDim3D = (m_physicsEngine->dimension() == Geometry::Dimension::dim3D);
  if (ImGui::Checkbox("3D", &isSystemDim3D))
  {
    m_physicsEngine->setDimension(isSystemDim3D ? Geometry::Dimension::dim3D : Geometry::Dimension::dim2D);
    m_graphicsEngine->setDimension(isSystemDim3D ? Geometry::Dimension::dim3D : Geometry::Dimension::dim2D);
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

  selectPhysicalQuantity();

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::SliderInt("Target FPS", &m_targetFps, 1, 60);

  ImGui::Text(" %.3f ms/frame (%.1f FPS) ", 1000.0f / m_currFps, m_currFps);

// Apple is not very OpenCL friendly
#ifndef __APPLE__
  bool isProfiling = m_physicsEngine->isProfilingEnabled();
  if (ImGui::Checkbox(" GPU Solver Profiling ", &isProfiling))
  {
    m_physicsEngine->enableProfiling(isProfiling);
  }
#endif

  ImGui::End();
}

bool ParticleSystemApp::selectPhysicalModel()
{
  // Selection of the physical model
  const auto& selModelName = (Physics::ALL_MODELS.find(m_modelType) != Physics::ALL_MODELS.end())
      ? Physics::ALL_MODELS.find(m_modelType)->second
      : Physics::ALL_MODELS.cbegin()->second;

  if (ImGui::BeginCombo("Physical Model", selModelName.c_str()))
  {
    for (const auto& model : Physics::ALL_MODELS)
    {
      if (ImGui::Selectable(model.second.c_str(), m_modelType == model.first))
      {
        m_modelType = model.first;

        if (!initGraphicsEngine())
        {
          LOG_ERROR("Failed to reset graphics engine");
          return false;
        }

        if (!initGraphicsWidget())
        {
          LOG_ERROR("Failed to reset graphics widget");
          return false;
        }

        if (!initPhysicsEngine())
        {
          LOG_ERROR("Failed to reset physics engine");
          return false;
        }

        if (!initPhysicsWidget())
        {
          LOG_ERROR("Failed to reset physics widget");
          return false;
        }

        LOG_INFO("Application correctly switched to {}", Physics::ALL_MODELS.find(m_modelType)->second);
      }
    }
    ImGui::EndCombo();
  }

  return true;
}

void ParticleSystemApp::selectPhysicalCase()
{
  // Current physical case
  auto selectedCaseType = m_physicsEngine->getCase();

  // Using json to do the mapping to the equivalent string
  std::string strCaseType = json(selectedCaseType);
  if (ImGui::BeginCombo("Physical case", strCaseType.c_str()))
  {
    auto beginCaseType = Utils::PhysicsCase::CASE_INVALID;
    auto endCaseType = Utils::PhysicsCase::CASE_INVALID;

    switch (m_modelType)
    {
    case Physics::BOIDS:
    {
      beginCaseType = Utils::PhysicsCase::BOIDS_BEGIN;
      endCaseType = Utils::PhysicsCase::BOIDS_END;
      break;
    }
    case Physics::FLUIDS:
    {
      beginCaseType = Utils::PhysicsCase::FLUIDS_BEGIN;
      endCaseType = Utils::PhysicsCase::FLUIDS_END;
      break;
    }
    case Physics::CLOUDS:
    {
      beginCaseType = Utils::PhysicsCase::CLOUDS_BEGIN;
      endCaseType = Utils::PhysicsCase::CLOUDS_END;
      break;
    }
    }

    for (int iCaseType = beginCaseType + 1; iCaseType != endCaseType; iCaseType++)
    {
      auto caseType = static_cast<Utils::PhysicsCase>(iCaseType);
      strCaseType = json(caseType);

      if (ImGui::Selectable(strCaseType.c_str(), selectedCaseType == caseType))
      {
        selectedCaseType = caseType;

        m_physicsEngine->setCase(selectedCaseType);
        m_physicsEngine->reset();
      }
    }
    ImGui::EndCombo();
  }
}

void ParticleSystemApp::selectPhysicalQuantity()
{
  // Selection of the physical quantity to render through particles intensity color (will fill color buffer used by fragment shader)
  if (m_physicsEngine->cbeginDisplayablePhysicalQuantities() != m_physicsEngine->cendDisplayablePhysicalQuantities())
  {
    const auto& selDisplayedQuantityName = m_physicsEngine->currentDisplayedPhysicalQuantityName();

    if (ImGui::BeginCombo("Physical Quantity", selDisplayedQuantityName.c_str()))
    {
      for (auto it = m_physicsEngine->beginDisplayablePhysicalQuantities(); it != m_physicsEngine->endDisplayablePhysicalQuantities(); ++it)
      {
        if (ImGui::Selectable(it->first.c_str(), selDisplayedQuantityName == it->first))
        {
          m_physicsEngine->setCurrentDisplayedQuantity(it->first);

          LOG_INFO("Visible physics quantity switched to {}", it->first);
        }
      }
      ImGui::EndCombo();
    }

    auto& currentQuantity = m_physicsEngine->currentDisplayedPhysicalQuantity();

    float sMin = currentQuantity.staticRange.first;
    float sMax = currentQuantity.staticRange.second;
    float uMin = currentQuantity.userRange.first;
    float uMax = currentQuantity.userRange.second;
    float eps = 0.001f; // To avoid empty range if two sliders values meet each other
    const std::string minStr = "Min [ " + Utils::FloatToStr(sMin, 2) + ", " + Utils::FloatToStr(uMax, 2) + " [";
    ImGui::SliderFloat(minStr.c_str(), &currentQuantity.userRange.first, sMin, uMax - eps, "%.3f");
    const std::string maxStr = "Max ] " + Utils::FloatToStr(uMin, 2) + ", " + Utils::FloatToStr(sMax, 2) + " ]";
    ImGui::SliderFloat(maxStr.c_str(), &currentQuantity.userRange.second, uMin + eps, sMax, "%.3f");
  }
}

bool ParticleSystemApp::popUpMessage(const std::string& title, const std::string& message) const
{
  bool closePopUp = false;

  bool open = true;
  ImGui::OpenPopup(title.c_str());
  if (ImGui::BeginPopupModal(title.c_str(), &open))
  {
    ImGui::Text(message.c_str(), "%s");
    if (ImGui::Button("Close"))
    {
      ImGui::CloseCurrentPopup();
      closePopUp = true;
    }
    ImGui::EndPopup();
  }

  return closePopUp;
}

} // End namespace App

int main(int, char**)
{
  Utils::InitializeLogger();

  App::ParticleSystemApp app;

  if (app.isInit())
  {
    app.run();
  }

  return 0;
}