
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
        boidsEngine->resetParticle(Core::Dimension::dim2D);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset 3D"))
    {
        boidsEngine->resetParticle(Core::Dimension::dim3D);
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
        if (ImGui::SliderFloat("Maximum Steering", &maxSteering, 0.005f, 2.0f))
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
        ImGui::Text(" Boids Behavior");
        ImGui::Spacing();
        bool isAlignment = boidsEngine->getActivateAlignment();
        if (ImGui::Checkbox("Alignment", &isAlignment))
        {
            boidsEngine->setActivateAlignment(isAlignment);
        }
        if (isAlignment)
        {
            float scaleAlignment = boidsEngine->getScaleAlignment();
            float radiusAlignment = boidsEngine->getRadiusAlignment();
            if (ImGui::SliderFloat("Scale Alignment", &scaleAlignment, 0.0, 5.0f))
            {
                boidsEngine->setScaleAlignment(scaleAlignment);
            }
            if (ImGui::SliderFloat("Radius Alignment", &radiusAlignment, 10.0f, 100.0f))
            {
                boidsEngine->setRadiusAlignment(radiusAlignment);
            }
        }
        ImGui::Spacing();
        bool isCohesion = boidsEngine->getActivateCohesion();
        if (ImGui::Checkbox("Cohesion", &isCohesion))
        {
            boidsEngine->setActivateCohesion(isCohesion);
        }
        if (isCohesion)
        {
            float scaleCohesion = boidsEngine->getScaleCohesion();
            float radiusCohesion = boidsEngine->getRadiusCohesion();
            if (ImGui::SliderFloat("Scale Cohesion", &scaleCohesion, 0.0f, 5.0f))
            {
                boidsEngine->setScaleCohesion(scaleCohesion);
            }
            if (ImGui::SliderFloat("Radius Cohesion", &radiusCohesion, 10.0f, 100.0f))
            {
                boidsEngine->setRadiusCohesion(radiusCohesion);
            }
        }
        bool isSeparation = boidsEngine->getActivateSeparation();
        if (ImGui::Checkbox("Separation", &isSeparation))
        {
            boidsEngine->setActivateSeparation(isSeparation);
        }
                if (isSeparation)
        {
            float scaleSeparation = boidsEngine->getScaleSeparation();
            float radiusSeparation = boidsEngine->getRadiusSeparation();
            if (ImGui::SliderFloat("Scale Separation", &scaleSeparation, 0.0f, 5.0f))
            {
                boidsEngine->setScaleSeparation(scaleSeparation);
            }
            if (ImGui::SliderFloat("Radius Separation", &radiusSeparation, 10.0f, 100.0f))
            {
                boidsEngine->setRadiusSeparation(radiusSeparation);
            }
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
