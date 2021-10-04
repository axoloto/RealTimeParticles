
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
  explicit PhysicsWidget(Physics::Model* physicsEngine)
      : m_physicsEngine(physicsEngine) {};
  virtual ~PhysicsWidget() = default;

  void display();

  private:
  void displayBoidsParameters(Physics::Boids* boids);
  void displayFluidsParameters(Physics::Fluids* engine);
  void displayBoundaryConditions(Physics::Model* engine);

  Physics::Model* m_physicsEngine;
};
}