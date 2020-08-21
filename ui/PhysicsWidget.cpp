
#include "imgui/imgui.h"
#include "PhysicsWidget.hpp"
#include "Boids.hpp"




UI::BoidsWidget::BoidsWidget(std::shared_ptr<Core::Physics> physicsEngine ) : PhysicsWidget(physicsEngine) 
{
    auto* boidsEngine = dynamic_cast<Core::Boids*>(m_physicsEngine.get());
    m_maxVelocity=boidsEngine->getmaxVelocity();
    m_maxSteering=boidsEngine->getmaxSteering();
    m_bouncingWall=boidsEngine->getbouncingWall();
    m_steering=boidsEngine->getSteering();
    m_pause=boidsEngine->getpause();
    m_forcedMaxSpeed=boidsEngine->getForcedMaxspeed();
};

void UI::BoidsWidget::display()
{
    auto* boidsEngine = dynamic_cast<Core::Boids*>(m_physicsEngine.get());

    if(!boidsEngine) return;
    
    ImGui::Begin("Boids Widget");

    if(ImGui::Checkbox("Pause System",&m_pause)){
        boidsEngine->setPause(m_pause);
    }
    ImGui::SameLine();
    if(ImGui::Button("Reset 2D")){  
        boidsEngine->resetBoids2D();        
    }
    ImGui::SameLine();
    if(ImGui::Button("Reset 3D")){  
        boidsEngine->resetBoids3D();            
    }
    ImGui::Spacing();
    if(ImGui::SliderFloat("Maximum Velocity", &m_maxVelocity, 0.01f, 20.0f))
    {
        boidsEngine->setmaxVelocity(m_maxVelocity);
    }
    if(ImGui::Checkbox("Force System to Max Speed",&m_forcedMaxSpeed)){
        boidsEngine->setForcedMaxSpeed(m_forcedMaxSpeed);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if(ImGui::Checkbox("Steering",&m_steering)){
        boidsEngine->setSteering(m_steering);
    }
    if(m_steering){
    if(ImGui::SliderFloat("Maximum Steering", &m_maxSteering, 0.01f, 20.0f))
    {
        boidsEngine->setmaxSteering(m_maxSteering);
    }
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Text(" Walls Behavior");
    if(ImGui::Checkbox("Wall",&m_bouncingWall)){
        boidsEngine->setbouncingWall(m_bouncingWall);
    }

    ImGui::End();
}
