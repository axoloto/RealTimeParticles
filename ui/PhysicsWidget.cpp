
#include "imgui/imgui.h"
#include "PhysicsWidget.hpp"
#include "Boids.hpp"

void UI::BoidsWidget::display()
{
    auto *boidsEngine = dynamic_cast<Core::Boids *>(m_physicsEngine.get());

    if (!boidsEngine)
        return;

    ImGui::Begin("Boids Widget");

    /*bool isPaused = boidsEngine->getPause();
    if (ImGui::Checkbox("Pause System", &isPaused))
    {
        boidsEngine->setPause(isPaused);
    }*/

    bool isPaused = boidsEngine->getPause();
    if (isPaused)
    {

        if (ImGui::Button("Run"))
        {
            boidsEngine->setPause(!isPaused);
        }
    }
    else
    {
        if (ImGui::Button("Pause"))
        {
            boidsEngine->setPause(!isPaused);
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset 2D"))
    {
        boidsEngine->resetBoids(Core::Dimension::dim2D);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset 3D"))
    {
        boidsEngine->resetBoids(Core::Dimension::dim3D);
    }
    ImGui::Spacing();
    float maxVelocity = boidsEngine->getmaxVelocity();
    if (ImGui::SliderFloat("Maximum Velocity", &maxVelocity, 0.01f, 20.0f))
    {
        boidsEngine->setmaxVelocity(maxVelocity);
    }
    bool isForcedmaxSpeed = boidsEngine->getForcedMaxspeed();
    if (ImGui::Checkbox("Force System to Max Speed", &isForcedmaxSpeed))
    {
        boidsEngine->setForcedMaxSpeed(isForcedmaxSpeed);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    bool isSteering = boidsEngine->getSteering();
    if (ImGui::Checkbox("Steering", &isSteering))
    {
        boidsEngine->setSteering(isSteering);
    }
    if (isSteering)
    {
        float maxSteering = boidsEngine->getmaxSteering();
        if (ImGui::SliderFloat("Maximum Steering", &maxSteering, 0.01f, 20.0f))
        {
            boidsEngine->setmaxSteering(maxSteering);
        }
        ImGui::Spacing();
        bool isTarget = boidsEngine->getActivateTargets();
        if (ImGui::Checkbox("Targets", &isTarget))
        {
            boidsEngine->setActivateTargets(isTarget);
        }
        ImGui::Spacing();
        bool isAlignment = boidsEngine->getActivateAlignment();
        if (ImGui::Checkbox("Alignment", &isAlignment))
        {
            boidsEngine->setActivateAlignment(isAlignment);
        }
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Text(" Walls Behavior");
    bool isBouncingWall = boidsEngine->getBouncingWall();
    bool isCyclicWall = boidsEngine->getCyclicWall();
    if (ImGui::Checkbox("BouncingWall", &isBouncingWall))
    {
        isCyclicWall = false;
        boidsEngine->setBouncingWall(isBouncingWall);
        boidsEngine->setCyclicWall(isCyclicWall);
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("CyclicWall", &isCyclicWall))
    {
        isBouncingWall = false;
        boidsEngine->setBouncingWall(isBouncingWall);
        boidsEngine->setCyclicWall(isCyclicWall);
    }

    ImGui::End();
}
