
#pragma once

#include "Engine.hpp"

#include <memory>

namespace UI
{
class GraphicsWidget
{
  public:
  explicit GraphicsWidget(Render::Engine* graphicsEngine)
      : m_graphicsEngine(graphicsEngine) {};
  virtual ~GraphicsWidget() = default;

  void display();

  private:
  Render::Engine* m_graphicsEngine;
};
}