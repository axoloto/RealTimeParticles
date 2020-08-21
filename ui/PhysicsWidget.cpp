
#include "imgui/imgui.h"
#include "PhysicsWidget.hpp"
#include "Boids.hpp"




UI::BoidsWidget::BoidsWidget(std::shared_ptr<Core::Physics> physicsEngine ) : PhysicsWidget(physicsEngine) 
{
    auto* boidsEngine = dynamic_cast<Core::Boids*>(m_physicsEngine.get());
    m_maxVelocity=boidsEngine->getmaxVelocity();
    m_bouncingWall=boidsEngine->getbouncingWall();
};

void UI::BoidsWidget::display()
{
    auto* boidsEngine = dynamic_cast<Core::Boids*>(m_physicsEngine.get());

    if(!boidsEngine) return;
    
    ImGui::Begin("Boids Widget");
    if(ImGui::SliderFloat("Maximum Velocity", &m_maxVelocity, 0.01f, 20.0f))
    {
        boidsEngine->setmaxVelocity(m_maxVelocity);
    }
    ImGui::Spacing;
    if(ImGui::Checkbox("Wall",&m_bouncingWall)){
        boidsEngine->setbouncingWall(m_bouncingWall);
    }

    ImGui::End();
}
