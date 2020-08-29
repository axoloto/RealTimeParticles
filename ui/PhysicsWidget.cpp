
#include "imgui/imgui.h"
#include "PhysicsWidget.hpp"
#include "Boids.hpp"

void UI::BoidsWidget::display()
{
    auto *boidsEngine = dynamic_cast<Core::Boids *>(m_physicsEngine.get());

    if (!boidsEngine) return;

    ImGui::Begin("Boids Widget", NULL, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PushItemWidth(150);

    bool isSteering = boidsEngine->getSteering();
    if (ImGui::Checkbox("Steering", &isSteering))
    {
        boidsEngine->setSteering(isSteering);
    }

    if (isSteering)
    {
        float maxSteering = boidsEngine->getmaxSteering();
        if (ImGui::SliderFloat("Max. Steering", &maxSteering, 0.005f, 2.0f))
        {
            boidsEngine->setMaxSteering(maxSteering);
        }

       // ImGui::Spacing();
        
       // bool isTarget = boidsEngine->getActivateTargets();
       // if (ImGui::Checkbox("Targets", &isTarget))
       // {
       //    boidsEngine->setActivateTargets(isTarget);
       // }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        bool isAlignment = boidsEngine->getActivateAlignment();
        if (ImGui::Checkbox("Alignment", &isAlignment))
        {
            boidsEngine->setActivateAlignment(isAlignment);
        }
        if (isAlignment)
        {
            ImGui::PushItemWidth(75);

            float scaleAlignment = boidsEngine->getScaleAlignment();
            if (ImGui::SliderFloat("##scaleAlign", &scaleAlignment, 0.0, 5.0f))
            {
                boidsEngine->setScaleAlignment(scaleAlignment);
            }
            ImGui::SameLine();
            float radiusAlignment = boidsEngine->getRadiusAlignment();
            if (ImGui::SliderFloat("##radAlign", &radiusAlignment, 10.0f, 100.0f))
            {
                boidsEngine->setRadiusAlignment(radiusAlignment);
            }

            ImGui::PushItemWidth(150);
        }

        bool isCohesion = boidsEngine->getActivateCohesion();
        if (ImGui::Checkbox("Cohesion", &isCohesion))
        {
            boidsEngine->setActivateCohesion(isCohesion);
        }
        if (isCohesion)
        {
            ImGui::PushItemWidth(75);

            float scaleCohesion = boidsEngine->getScaleCohesion();
            if (ImGui::SliderFloat("##scaleCoh", &scaleCohesion, 0.0f, 5.0f))
            {
                boidsEngine->setScaleCohesion(scaleCohesion);
            }
            ImGui::SameLine();
            float radiusCohesion = boidsEngine->getRadiusCohesion();
            if (ImGui::SliderFloat("##radCoh", &radiusCohesion, 10.0f, 100.0f))
            {
                boidsEngine->setRadiusCohesion(radiusCohesion);
            }

            ImGui::PushItemWidth(150);
        }

        bool isSeparation = boidsEngine->getActivateSeparation();
        if (ImGui::Checkbox("Separation", &isSeparation))
        {
            boidsEngine->setActivateSeparation(isSeparation);
        }
        if (isSeparation)
        {
            ImGui::PushItemWidth(75);

            float scaleSeparation = boidsEngine->getScaleSeparation();
            if (ImGui::SliderFloat("##scaleSep", &scaleSeparation, 0.0f, 5.0f))
            {
                boidsEngine->setScaleSeparation(scaleSeparation);
            }
            ImGui::SameLine();
            float radiusSeparation = boidsEngine->getRadiusSeparation();
            if (ImGui::SliderFloat("##radSep", &radiusSeparation, 10.0f, 100.0f))
            {
                boidsEngine->setRadiusSeparation(radiusSeparation);
            }

            ImGui::PushItemWidth(150);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text(" Wall Behavior");
    bool isBouncingWall = boidsEngine->getBouncingWall();
    if (ImGui::Checkbox("Bouncing Wall", &isBouncingWall))
    {
        boidsEngine->setBouncingWall(isBouncingWall);
        boidsEngine->setCyclicWall(!isBouncingWall);
    }

    ImGui::SameLine();

    bool isCyclicWall = boidsEngine->getCyclicWall();
    if (ImGui::Checkbox("Cyclic Wall", &isCyclicWall))
    {
        boidsEngine->setBouncingWall(!isCyclicWall);
        boidsEngine->setCyclicWall(isCyclicWall);
    }

    ImGui::End();
}
