
#pragma once

#include "Model.hpp"

#include <memory>

namespace UI
{
class PhysicsWidget
{
  public:
  explicit PhysicsWidget(std::shared_ptr<Physics::Model> physicsEngine)
      : m_physicsEngine(physicsEngine) {};
  virtual ~PhysicsWidget() = default;

  void display();

  private:
  std::weak_ptr<Physics::Model> m_physicsEngine;
};
}