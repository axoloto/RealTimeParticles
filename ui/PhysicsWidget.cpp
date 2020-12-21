
#include "Boids.hpp"
#include "PhysicsWidget.hpp"
#include <imgui.h>

void UI::BoidsWidget::display()
{
  auto& boidsEngine = dynamic_cast<Core::Boids&>(m_physicsEngine);

  ImGui::Begin("Boids Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
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

  ImGui::Text(" Boundary ");
  bool isBouncingWall = (boidsEngine.boundary() == Core::Boundary::BouncingWall);
  if (ImGui::Checkbox("Bouncing Wall", &isBouncingWall))
  {
    if (isBouncingWall)
      boidsEngine.setBoundary(Core::Boundary::BouncingWall);
    else
      boidsEngine.setBoundary(Core::Boundary::CyclicWall);
  }

  ImGui::SameLine();

  bool isCyclicWall = (boidsEngine.boundary() == Core::Boundary::CyclicWall);
  if (ImGui::Checkbox("Cyclic Wall", &isCyclicWall))
  {
    if (isCyclicWall)
      boidsEngine.setBoundary(Core::Boundary::CyclicWall);
    else
      boidsEngine.setBoundary(Core::Boundary::BouncingWall);
  }

  ImGui::End();
}