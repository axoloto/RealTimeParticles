#include "GraphicsWidget.hpp"
#include "Logging.hpp"
#include "Parameters.hpp"

#include <imgui.h>

void UI::GraphicsWidget::display()
{
  auto* graphicsEngine = dynamic_cast<Render::Engine*>(m_graphicsEngine);

  if (!graphicsEngine)
    return;

  // First default pos
  ImGui::SetNextWindowPos(ImVec2(60, 830), ImGuiCond_FirstUseEver);

  ImGui::Begin("Graphics Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  int pointSize = (int)graphicsEngine->getPointSize();
  if (ImGui::SliderInt("Particle size", &pointSize, 1, 5))
  {
    graphicsEngine->setPointSize((size_t)pointSize);
  }
  ImGui::End();
}