
#include "imgui/imgui.h"
#include "PhysicsWidget.hpp"
#include "Boids.hpp"


void UI::BoidsWidget::display()
{
    auto* boidsEngine = dynamic_cast<Core::Physics*>(m_physicsEngine.get());

    if(!boidsEngine) return;

    ImGui::Begin("Boids Widget");


    ImGui::End();
}
