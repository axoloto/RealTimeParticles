#include "PhysicsWidget.hpp"
#include "Logging.hpp"
#include "Parameters.hpp"
#include <imgui.h>
#include <vector>

void displayBoundaryConditions(Physics::Model* engine)
{
  if (!engine)
    return;

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::Text(" Boundary ");
  ImGui::Spacing();

  bool isBouncingWall = (engine->boundary() == Physics::Boundary::BouncingWall);
  if (ImGui::Checkbox("Bouncing Wall", &isBouncingWall))
  {
    if (isBouncingWall)
      engine->setBoundary(Physics::Boundary::BouncingWall);
    else
      engine->setBoundary(Physics::Boundary::CyclicWall);
  }

  ImGui::SameLine();

  bool isCyclicWall = (engine->boundary() == Physics::Boundary::CyclicWall);
  if (ImGui::Checkbox("Cyclic Wall", &isCyclicWall))
  {
    if (isCyclicWall)
      engine->setBoundary(Physics::Boundary::CyclicWall);
    else
      engine->setBoundary(Physics::Boundary::BouncingWall);
  }
}

void drawImguiCheckBoxFromJson(const std::string& name, bool& enable)
{
  ImGui::Checkbox(name.c_str(), &enable);
}

void drawImguiSliderInt(const std::string& name, json& js)
{
  int intVal = js.at(0);
  int minVal = js.at(1);
  int maxVal = js.at(2);
  if (ImGui::SliderInt(name.c_str(), &intVal, minVal, maxVal))
  {
    js.at(0) = intVal;
  }
}

void drawImguiSliderFloat(const std::string& name, json& js)
{
  float floatVal = js.at(0);
  float minVal = js.at(1);
  float maxVal = js.at(2);
  std::string precision = floatVal <= 0.1f ? "%.4f" : "%.2f";
  if (ImGui::SliderFloat(name.c_str(), &floatVal, minVal, maxVal, precision.c_str()))
  {
    js.at(0) = floatVal;
  }
}

template <typename EnumType>
void drawImguiEnumCombo(const std::string& name, json& js)
{
  if (ImGui::BeginCombo(name.c_str(), js.at(0).get<std::string>().c_str()))
  {
    int min = static_cast<int>(js.at(1).get<EnumType>());
    int max = static_cast<int>(js.at(2).get<EnumType>());

    // we skip min and max values, only here as boundaries
    for (int caseIndex = min + 1; caseIndex < max; ++caseIndex)
    {
      // int -> enum conversion
      auto caseT = static_cast<EnumType>(caseIndex);
      // Using json to do the enum -> string conversion
      json selJs = caseT;

      if (ImGui::Selectable(selJs.get<std::string>().c_str(), js.at(0).get<EnumType>() == caseT))
      {
        js.at(0) = caseT;
      }
    }
    ImGui::EndCombo();
  }
}

void drawImguiObjectFromJson(json& js)
{
  for (auto& el : js.items())
  {
    auto& val = el.value();
    if (val.is_object())
    {
      ImGui::Spacing();
      ImGui::Text(el.key().c_str());
      ImGui::Indent(15.0f);
      // Recursive call
      drawImguiObjectFromJson(val);
      ImGui::Unindent(15.0f);
      ImGui::Spacing();
    }
    else if (val.is_boolean())
    {
      drawImguiCheckBoxFromJson(el.key(), val.get_ref<bool&>());

      // special case where we skip the rest of the items if "Enable" param is false
      bool skipRestOfItems = el.key().find("Enable##") != std::string::npos && val == false;

      if (skipRestOfItems)
        return;
    }
    else if (val.is_array() && val[0].get<Utils::PhysicsCase>() != Utils::PhysicsCase::CASE_INVALID)
    {
      drawImguiEnumCombo<Utils::PhysicsCase>(el.key(), val);
    }
    else if (val.is_array() && val.size() == 3 && val[0].is_number_integer())
    {
      // cannot directly access json array items by reference
      drawImguiSliderInt(el.key(), val);
    }
    else if (val.is_array() && val.size() == 3 && val[0].is_number_float())
    {
      // cannot directly access json array items by reference
      drawImguiSliderFloat(el.key(), val);
    }
  }
}

void UI::PhysicsWidget::display()
{
  auto physicsEngine = m_physicsEngine.lock();

  if (!physicsEngine)
    return;

  // First default pos
  ImGui::SetNextWindowPos(ImVec2(15, 355), ImGuiCond_FirstUseEver);

  ImGui::Value("Particles", (int)physicsEngine->nbParticles());

  // Retrieve input json from the physics engine with all available parameters
  json js = physicsEngine->getInputJson();
  // Draw all items from input json
  drawImguiObjectFromJson(js);
  // Update physics engine with new parameters values if any
  physicsEngine->updateInputJson(js);
}