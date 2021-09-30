
#pragma once

#include "Engine.hpp"
#include "GraphicsWidget.hpp"
#include "Math.hpp"
#include "Model.hpp"
#include "Parameters.hpp"
#include "PhysicsWidget.hpp"
#include <SDL.h>
#include <imgui.h>


namespace App
{
class ParticleSystemApp
{
  public:
  ParticleSystemApp();
  ~ParticleSystemApp() = default;
  void run();
  bool isInit() const { return m_init; }

  private:
  bool initWindow();
  bool initGraphicsEngine();
  bool initPhysicsEngine();
  bool initPhysicsWidget();
  bool initGraphicsWidget();
  bool closeWindow();
  bool checkSDLStatus();
  void checkMouseState();
  void displayMainWidget();
  bool popUpErrorMessage(std::string errorMessage);

  std::unique_ptr<Physics::Model> m_physicsEngine;
  std::unique_ptr<Render::Engine> m_graphicsEngine;
  std::unique_ptr<UI::PhysicsWidget> m_physicsWidget;
  std::unique_ptr<UI::GraphicsWidget> m_graphicsWidget;

  SDL_Window* m_window;
  SDL_GLContext m_OGLContext;

  std::string m_nameApp;

  Physics::ModelType m_modelType;

  Math::int2 m_windowSize;
  Math::int2 m_mousePrevPos;
  ImVec4 m_backGroundColor;
  bool m_buttonLeftActivated;
  bool m_buttonRightActivated;
  bool m_init;
};

}