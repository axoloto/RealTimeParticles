
#pragma once

#include "Physics.hpp"
#include <memory>

namespace UI
{
class PhysicsWidget
{
  public:
  explicit PhysicsWidget(Core::Physics& physicsEngine)
      : m_physicsEngine(physicsEngine) {};
  virtual ~PhysicsWidget() = default;

  virtual void display() = 0;

  protected:
  Core::Physics& m_physicsEngine;
};
}