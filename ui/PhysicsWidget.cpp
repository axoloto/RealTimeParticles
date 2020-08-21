
#include "imgui/imgui.h"
#include "PhysicsWidget.hpp"
#include "Boids.hpp"




void UI::BoidsWidget::display()
{
    auto* boidsEngine = dynamic_cast<Core::Boids*>(m_physicsEngine.get());

    if(!boidsEngine) return;
    
    ImGui::Begin("Boids Widget");

    bool isPaused = boidsEngine->getPause();
    if(ImGui::Checkbox("Pause System",&isPaused)){
        boidsEngine->setPause(isPaused);
    }
    ImGui::SameLine();
    if(ImGui::Button("Reset 2D")){  
        boidsEngine->resetBoids(2);        
    }
    ImGui::SameLine();
    if(ImGui::Button("Reset 3D")){  
        boidsEngine->resetBoids(3);            
    }
    ImGui::Spacing();
    float maxVelocity = boidsEngine->getmaxVelocity();
    if(ImGui::SliderFloat("Maximum Velocity", &maxVelocity, 0.01f, 20.0f))
    {
        boidsEngine->setmaxVelocity(maxVelocity);
    }
    bool isForcedmaxSpeed = boidsEngine->getForcedMaxspeed();
    if(ImGui::Checkbox("Force System to Max Speed",&isForcedmaxSpeed)){
        boidsEngine->setForcedMaxSpeed(isForcedmaxSpeed);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    bool isSteering = boidsEngine->getSteering();
    if(ImGui::Checkbox("Steering",&isSteering)){
        boidsEngine->setSteering(isSteering);
    }
    if(isSteering){
    float maxSteering=boidsEngine->getmaxSteering();
    if(ImGui::SliderFloat("Maximum Steering", &maxSteering, 0.01f, 20.0f))
    {
        boidsEngine->setmaxSteering(maxSteering);
    }
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Text(" Walls Behavior");
    bool isBouncingWall = boidsEngine->getBouncingWall();
    if(ImGui::Checkbox("Wall",&isBouncingWall)){
        boidsEngine->setBouncingWall(isBouncingWall);
    }

    ImGui::End();
}
