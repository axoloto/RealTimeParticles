#include "PhysicsWidget.hpp"
#include "Logging.hpp"
#include "Parameters.hpp"

#include <imgui.h>

void UI::PhysicsWidget::display()
{
  auto* boidsEngine = dynamic_cast<Physics::Boids*>(m_physicsEngine);
  auto* fluidsEngine = dynamic_cast<Physics::Fluids*>(m_physicsEngine);

  // First default pos
  ImGui::SetNextWindowPos(ImVec2(15, 355), ImGuiCond_FirstUseEver);

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

  // Selection of the initial setup
  auto caseType = fluidsEngine->getInitialCase();

  const auto& selCaseName = (Physics::ALL_FLUID_CASES.find(caseType) != Physics::ALL_FLUID_CASES.end())
      ? Physics::ALL_FLUID_CASES.find(caseType)->second
      : Physics::ALL_FLUID_CASES.cbegin()->second;

  if (ImGui::BeginCombo("Study case", selCaseName.c_str()))
  {
    for (const auto& caseT : Physics::ALL_FLUID_CASES)
    {
      if (ImGui::Selectable(caseT.second.c_str(), caseType == caseT.first))
      {
        caseType = caseT.first;

        fluidsEngine->setInitialCase(caseType);
        fluidsEngine->reset();

        LOG_DEBUG("Fluids initial case correctly switched to {}", Physics::ALL_FLUID_CASES.find(caseType)->second);
      }
    }
    ImGui::EndCombo();
  }

  ImGui::Value("Particles", (int)fluidsEngine->nbParticles());

  ImGui::Spacing();
  ImGui::Text("Fluid parameters");
  ImGui::Spacing();

  ImGui::Value("Kernel radius", (float)fluidsEngine->getEffectRadius());

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
  if (ImGui::SliderFloat("Time Step", &timeStep, 0.0001f, 0.015f))
  {
    fluidsEngine->setTimeStep(timeStep);
  }

  int nbJacobiIters = (int)fluidsEngine->getNbJacobiIters();
  if (ImGui::SliderInt("Nb Jacobi Iterations", &nbJacobiIters, 1, 6))
  {
    fluidsEngine->setNbJacobiIters((size_t)nbJacobiIters);
  }

  bool isArtPressureEnabled = fluidsEngine->isArtPressureEnabled();
  if (ImGui::Checkbox("Enable Artificial Pressure", &isArtPressureEnabled))
  {
    fluidsEngine->enableArtPressure(isArtPressureEnabled);
  }
  if (isArtPressureEnabled)
  {
    float artPressureCoeff = fluidsEngine->getArtPressureCoeff();
    if (ImGui::SliderFloat("Coefficient", &artPressureCoeff, 0.0f, 0.001f, "%.4f"))
    {
      fluidsEngine->setArtPressureCoeff(artPressureCoeff);
    }

    float artPressureRadius = fluidsEngine->getArtPressureRadius();
    if (ImGui::SliderFloat("Radius", &artPressureRadius, 0.001f, 0.015f))
    {
      fluidsEngine->setArtPressureRadius(artPressureRadius);
    }

    int artPressureExp = (int)fluidsEngine->getArtPressureExp();
    if (ImGui::SliderInt("Exponent", &artPressureExp, 1, 6))
    {
      fluidsEngine->setArtPressureExp((size_t)artPressureExp);
    }
  }

  bool isVorticityConfinementEnabled = fluidsEngine->isVorticityConfinementEnabled();
  if (ImGui::Checkbox("Enable Vorticity Confinement", &isVorticityConfinementEnabled))
  {
    fluidsEngine->enableVorticityConfinement(isVorticityConfinementEnabled);
  }
  if (isVorticityConfinementEnabled)
  {
    float vorticityConfinementCoeff = fluidsEngine->getVorticityConfinementCoeff();
    if (ImGui::SliderFloat("Vorticity Coefficient", &vorticityConfinementCoeff, 0.0f, 0.001f, "%.4f"))
    {
      fluidsEngine->setVorticityConfinementCoeff(vorticityConfinementCoeff);
    }

    float xsphViscosityCoeff = fluidsEngine->getXsphViscosityCoeff();
    if (ImGui::SliderFloat("Viscosity Coefficient", &xsphViscosityCoeff, 0.0f, 0.001f, "%.4f"))
    {
      fluidsEngine->setXsphViscosityCoeff(xsphViscosityCoeff);
    }
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
  ImGui::Text("Speed");
  ImGui::Spacing();

  float velocity = boidsEngine->velocity();
  if (ImGui::SliderFloat("##speed", &velocity, 0.01f, 5.0f))
  {
    boidsEngine->setVelocity(velocity);
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Text("Particles");
  ImGui::Spacing();

  // Selection of the number of particles in the model
  const auto nbParticles = (Utils::NbParticles)m_physicsEngine->nbParticles();

  const auto& nbParticlesStr = (Utils::ALL_NB_PARTICLES.find(nbParticles) != Utils::ALL_NB_PARTICLES.end())
      ? Utils::ALL_NB_PARTICLES.find(nbParticles)->second.name
      : Utils::ALL_NB_PARTICLES.cbegin()->second.name;

  if (ImGui::BeginCombo("##particles", nbParticlesStr.c_str()))
  {
    for (const auto& nbParticlesPair : Utils::ALL_NB_PARTICLES)
    {
      if (ImGui::Selectable(nbParticlesPair.second.name.c_str(), nbParticles == nbParticlesPair.first))
      {
        boidsEngine->setNbParticles(nbParticlesPair.first);
        boidsEngine->reset();
      }
    }
    ImGui::EndCombo();
  }

  ImGui::Spacing();
  ImGui::Separator();
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
    if (ImGui::SliderFloat("##targetRadius", &targetRadius, 1.0f, 20.0f))
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
