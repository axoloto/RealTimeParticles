
#pragma once

#include "Boids.hpp"
#include "Fluids.hpp"
#include "Physics.hpp"

#include <memory>

namespace UI
{
class PhysicsWidget
{
  public:
  explicit PhysicsWidget(std::shared_ptr<Core::Physics> physicsEngine)
      : m_physicsEngine(physicsEngine) {};
  virtual ~PhysicsWidget() = default;

  void display();

  private:
  void displayBoidsParameters(Core::Boids* boids);
  void displayFluidsParameters(Core::Fluids* engine);

  std::shared_ptr<Core::Physics> m_physicsEngine;
};
}