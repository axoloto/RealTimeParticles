#include "GraphicsWidget.hpp"
#include "Logging.hpp"
#include "Parameters.hpp"

#include <imgui.h>

void UI::GraphicsWidget::display()
{
  if (!m_graphicsEngine)
    return;

  // First default pos
  ImGui::SetNextWindowPos(ImVec2(15, 192), ImGuiCond_FirstUseEver);

  ImGui::Begin("Graphics Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  const auto cameraPos = m_graphicsEngine->cameraPos();
  const auto focusPos = m_graphicsEngine->focusPos();

  ImGui::Text(" Camera (%.1f, %.1f, %.1f)", cameraPos.x, cameraPos.y, cameraPos.z);
  ImGui::Text(" Target (%.1f, %.1f, %.1f)", focusPos.x, focusPos.y, focusPos.z);
  ImGui::Text(" Dist. camera target : %.1f", Math::length(cameraPos - focusPos));
  ImGui::Spacing();

  bool isAutoRotating = m_graphicsEngine->isCameraAutoRotating();
  if (ImGui::Checkbox(" Auto rotation ", &isAutoRotating))
  {
    m_graphicsEngine->autoRotateCamera(isAutoRotating);
  }

  if (ImGui::Button(" Reset Camera "))
  {
    m_graphicsEngine->resetCamera();
  }

  int pointSize = (int)m_graphicsEngine->getPointSize();
  if (ImGui::SliderInt("Particle size", &pointSize, 1, 10))
  {
    m_graphicsEngine->setPointSize((size_t)pointSize);
  }
  ImGui::End();
}