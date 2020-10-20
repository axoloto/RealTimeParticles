
#include "PhysicsWidget.hpp"
#include "imgui/imgui.h"

#ifndef OPENCL_ACTIVATED
void UI::OCLBoidsWidget::display() {}
#else
#include "ocl/OCLBoids.hpp"

void UI::OCLBoidsWidget::display()
{
  auto& boidsEngine = dynamic_cast<Core::OCLBoids&>(m_physicsEngine);

  ImGui::Begin("OCL Boids Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  ImGui::Spacing();

  bool isTarget = boidsEngine.isTargetActivated();
  if (ImGui::Checkbox("Center Target", &isTarget))
  {
    boidsEngine.activateTarget(isTarget);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  bool isAlignment = boidsEngine.isAlignmentActivated();
  if (ImGui::Checkbox("Alignment", &isAlignment))
  {
    boidsEngine.activateAlignment(isAlignment);
  }
  if (isAlignment)
  {
    ImGui::PushItemWidth(75);

    float scaleAlignment = boidsEngine.scaleAlignment();
    if (ImGui::SliderFloat("##scaleAlign", &scaleAlignment, 0.0, 5.0f))
    {
      boidsEngine.setScaleAlignment(scaleAlignment);
    }

    ImGui::PushItemWidth(150);
  }

  bool isCohesion = boidsEngine.isCohesionActivated();
  if (ImGui::Checkbox("Cohesion", &isCohesion))
  {
    boidsEngine.activateCohesion(isCohesion);
  }
  if (isCohesion)
  {
    ImGui::PushItemWidth(75);

    float scaleCohesion = boidsEngine.scaleCohesion();
    if (ImGui::SliderFloat("##scaleCoh", &scaleCohesion, 0.0f, 5.0f))
    {
      boidsEngine.setScaleCohesion(scaleCohesion);
    }

    ImGui::PushItemWidth(150);
  }

  bool isSeparation = boidsEngine.isSeparationActivated();
  if (ImGui::Checkbox("Separation", &isSeparation))
  {
    boidsEngine.activateSeparation(isSeparation);
  }
  if (isSeparation)
  {
    ImGui::PushItemWidth(75);

    float scaleSeparation = boidsEngine.scaleSeparation();
    if (ImGui::SliderFloat("##scaleSep", &scaleSeparation, 0.0f, 5.0f))
    {
      boidsEngine.setScaleSeparation(scaleSeparation);
    }

    ImGui::PushItemWidth(150);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::Text(" Wall Behavior");
  bool isBouncingWall = boidsEngine.isBouncingWallEnabled();
  if (ImGui::Checkbox("Bouncing Wall", &isBouncingWall))
  {
    boidsEngine.setBouncingWall(isBouncingWall);
    boidsEngine.setCyclicWall(!isBouncingWall);
  }

  ImGui::SameLine();

  bool isCyclicWall = boidsEngine.isCyclicWallEnabled();
  if (ImGui::Checkbox("Cyclic Wall", &isCyclicWall))
  {
    boidsEngine.setBouncingWall(!isCyclicWall);
    boidsEngine.setCyclicWall(isCyclicWall);
  }

  ImGui::End();
}

#endif