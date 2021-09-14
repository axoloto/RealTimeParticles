
#pragma once

#include "Boids.hpp"
#include "Fluids.hpp"
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
  void displayBoidsParameters(Physics::Boids* boids);
  void displayFluidsParameters(Physics::Fluids* engine);

  std::shared_ptr<Physics::Model> m_physicsEngine;
};
}