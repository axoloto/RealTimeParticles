#include "PhysicsWidget.hpp"

#include <imgui.h>

void UI::PhysicsWidget::display()
{
  auto* boidsEngine = dynamic_cast<Physics::Boids*>(m_physicsEngine);
  auto* fluidsEngine = dynamic_cast<Physics::Fluids*>(m_physicsEngine);

  // First default pos
  ImGui::SetNextWindowPos(ImVec2(60, 430), ImGuiCond_FirstUseEver);

  if (boidsEngine)
    displayBoidsParameters(boidsEngine);
  else if (fluidsEngine)
    displayFluidsParameters(fluidsEngine);
}

void UI::PhysicsWidget::displayFluidsParameters(Physics::Fluids* fluidsEngine)
{
  if (!fluidsEngine)
    return;

  ImGui::Begin("Fluids Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  ImGui::Spacing();
  ImGui::Text("Cases");
  ImGui::Spacing();
  /*
  // Selection of the initial setup
  const auto& selCaseName = (Physics::Fluids::ALL_CASES.find(m_modelType) != Physics::ALL_CASES.end())
      ? Physics::ALL_CASES.find(m_modelType)->second
      : Physics::ALL_CASES.cbegin()->second;

  if (ImGui::BeginCombo("Physical Model", selCaseName.c_str()))
  {
    for (const auto& model : Physics::ALL_CASES)
    {
      if (ImGui::Selectable(model.second.c_str(), m_modelType == model.first))
      {
        m_modelType = model.first;

        if (!initPhysicsEngine())
        {
          LOG_ERROR("Failed to change physics engine");
          return;
        }

        if (!initPhysicsWidget())
        {
          LOG_ERROR("Failed to change physics widget");
          return;
        }

        m_physicsEngine->setNbParticles(m_nbParticles);
        m_graphicsEngine->setNbParticles(m_nbParticles);

        LOG_INFO("Application correctly switched to {}", Physics::ALL_CASES.find(m_modelType)->second);
      }
    }
    ImGui::EndCombo();
  }

*/
  ImGui::Spacing();
  ImGui::Text("Fluid parameters");
  ImGui::Spacing();

  float effectRadius = fluidsEngine->getEffectRadius();
  if (ImGui::SliderFloat("Effect Radius", &effectRadius, 0.01f, 0.9f))
  {
    fluidsEngine->setEffectRadius(effectRadius);
  }

  float restDensity = fluidsEngine->getRestDensity();
  if (ImGui::SliderFloat("Rest Density", &restDensity, 10.0f, 1000.0f))
  {
    fluidsEngine->setRestDensity(restDensity);
  }

  float relaxCFM = fluidsEngine->getRelaxCFM();
  if (ImGui::SliderFloat("Relax CFM", &relaxCFM, 100.0f, 1000.f))
  {
    fluidsEngine->setRelaxCFM(relaxCFM);
  }

  float timeStep = fluidsEngine->getTimeStep();
  if (ImGui::SliderFloat("Time Step", &timeStep, 0.0001f, 0.1f))
  {
    fluidsEngine->setTimeStep(timeStep);
  }

  ImGui::End();
}

void UI::PhysicsWidget::displayBoidsParameters(Physics::Boids* boidsEngine)
{
  if (!boidsEngine)
    return;

  ImGui::Begin("Boids Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  ImGui::Spacing();
  ImGui::Text("Target");
  ImGui::Spacing();

  bool isTarget = boidsEngine->isTargetActivated();
  if (ImGui::Checkbox("Activate", &isTarget))
  {
    boidsEngine->activateTarget(isTarget);
    boidsEngine->setTargetVisibility(isTarget);
  }

  if (isTarget)
  {
    bool isTargetVisible = boidsEngine->isTargetVisible();
    if (ImGui::Checkbox("Show", &isTargetVisible))
    {
      boidsEngine->setTargetVisibility(isTargetVisible);
    }

    float targetRadius = boidsEngine->targetRadiusEffect();
    if (ImGui::SliderFloat("##targetRadius", &targetRadius, 30.0f, 1000.0f))
    {
      boidsEngine->setTargetRadiusEffect(targetRadius);
    }

    int targetSign = boidsEngine->targetSignEffect();

    bool repulse = (targetSign < 0);
    if (ImGui::Checkbox("Repulse", &repulse))
    {
      boidsEngine->setTargetSignEffect(-1);
    }

    ImGui::SameLine();

    bool attract = (targetSign > 0);
    if (ImGui::Checkbox("Attract", &attract))
    {
      boidsEngine->setTargetSignEffect(1);
    }
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::Text("Boids Rules");
  ImGui::Spacing();

  bool isAlignment = boidsEngine->isAlignmentActivated();
  if (ImGui::Checkbox("Alignment", &isAlignment))
  {
    boidsEngine->activateAlignment(isAlignment);
  }
  if (isAlignment)
  {
    ImGui::PushItemWidth(75);

    float scaleAlignment = boidsEngine->scaleAlignment();
    if (ImGui::SliderFloat("##scaleAlign", &scaleAlignment, 0.0, 3.0f))
    {
      boidsEngine->setScaleAlignment(scaleAlignment);
    }

    ImGui::PushItemWidth(150);
  }

  bool isCohesion = boidsEngine->isCohesionActivated();
  if (ImGui::Checkbox("Cohesion", &isCohesion))
  {
    boidsEngine->activateCohesion(isCohesion);
  }
  if (isCohesion)
  {
    ImGui::PushItemWidth(75);

    float scaleCohesion = boidsEngine->scaleCohesion();
    if (ImGui::SliderFloat("##scaleCoh", &scaleCohesion, 0.0f, 3.0f))
    {
      boidsEngine->setScaleCohesion(scaleCohesion);
    }

    ImGui::PushItemWidth(150);
  }

  bool isSeparation = boidsEngine->isSeparationActivated();
  if (ImGui::Checkbox("Separation", &isSeparation))
  {
    boidsEngine->activateSeparation(isSeparation);
  }
  if (isSeparation)
  {
    ImGui::PushItemWidth(75);

    float scaleSeparation = boidsEngine->scaleSeparation();
    if (ImGui::SliderFloat("##scaleSep", &scaleSeparation, 0.0f, 3.0f))
    {
      boidsEngine->setScaleSeparation(scaleSeparation);
    }

    ImGui::PushItemWidth(150);
  }

  displayBoundaryConditions(boidsEngine);

  ImGui::End();
}

void UI::PhysicsWidget::displayBoundaryConditions(Physics::Model* engine)
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
