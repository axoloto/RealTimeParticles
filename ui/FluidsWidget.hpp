
#pragma once

#include "PhysicsWidget.hpp"
#include <memory>

namespace UI
{
class FluidsWidget : public PhysicsWidget
{
  public:
  explicit FluidsWidget(Core::Physics& physicsEngine)
      : PhysicsWidget(physicsEngine) {};
  ~FluidsWidget() override = default;
  void display() override;

  private:
};
}