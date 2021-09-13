#pragma once

#include "PhysicsWidget.hpp"

namespace UI
{
class BoidsWidget : public PhysicsWidget
{
  public:
  explicit BoidsWidget(Core::Physics& physicsEngine)
      : PhysicsWidget(physicsEngine) {};
  ~BoidsWidget() override = default;
  void display() override;

  private:
};
}