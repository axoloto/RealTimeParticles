#include "PhysicsWidget.hpp"
#include "Logging.hpp"
#include "Parameters.hpp"
#include <vector>
//#include "Boids.hpp"
//#include "Clouds.hpp"
//#include "Fluids.hpp"

#include <imgui.h>

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

/*
void displayCloudsParameters(Physics::Clouds* cloudsEngine)
{
  if (!cloudsEngine)
    return;

  ImGui::Begin("Clouds Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  // Selection of the initial setup
  auto caseType = cloudsEngine->getInitialCase();

  const auto& selCaseName = (Physics::Clouds::ALL_CASES.find(caseType) != Physics::Clouds::ALL_CASES.end())
      ? Physics::Clouds::ALL_CASES.find(caseType)->second
      : Physics::Clouds::ALL_CASES.cbegin()->second;

  if (ImGui::BeginCombo("Study case", selCaseName.c_str()))
  {
    for (const auto& caseT : Physics::Clouds::ALL_CASES)
    {
      if (ImGui::Selectable(caseT.second.c_str(), caseType == caseT.first))
      {
        caseType = caseT.first;

        cloudsEngine->setInitialCase(caseType);
        cloudsEngine->reset();

        LOG_DEBUG("Clouds initial case correctly switched to {}", Physics::Clouds::ALL_CASES.find(caseType)->second);
      }
    }
    ImGui::EndCombo();
  }

  ImGui::Value("Particles", (int)cloudsEngine->nbParticles());

  ImGui::Spacing();
  ImGui::Text("Position Based Fluids Parameters");
  ImGui::Spacing();

  float restDensity = cloudsEngine->getRestDensity();
  if (ImGui::SliderFloat("Rest Density", &restDensity, 10.0f, 1000.0f))
  {
    cloudsEngine->setRestDensity(restDensity);
  }

  float relaxCFM = cloudsEngine->getRelaxCFM();
  if (ImGui::SliderFloat("Relax CFM", &relaxCFM, 100.0f, 1000.f))
  {
    cloudsEngine->setRelaxCFM(relaxCFM);
  }

  float timeStep = cloudsEngine->getTimeStep();
  if (ImGui::SliderFloat("Time Step", &timeStep, 0.0001f, 0.020f))
  {
    cloudsEngine->setTimeStep(timeStep);
  }

  int nbJacobiIters = (int)cloudsEngine->getNbJacobiIters();
  if (ImGui::SliderInt("Nb Jacobi Iterations", &nbJacobiIters, 1, 6))
  {
    cloudsEngine->setNbJacobiIters((size_t)nbJacobiIters);
  }

  bool isArtPressureEnabled = cloudsEngine->isArtPressureEnabled();
  if (ImGui::Checkbox("Enable Artificial Pressure", &isArtPressureEnabled))
  {
    cloudsEngine->enableArtPressure(isArtPressureEnabled);
  }
  if (isArtPressureEnabled)
  {
    float artPressureCoeff = cloudsEngine->getArtPressureCoeff();
    if (ImGui::SliderFloat("Coefficient", &artPressureCoeff, 0.0f, 0.001f, "%.4f"))
    {
      cloudsEngine->setArtPressureCoeff(artPressureCoeff);
    }

    float artPressureRadius = cloudsEngine->getArtPressureRadius();
    if (ImGui::SliderFloat("Radius", &artPressureRadius, 0.001f, 0.015f))
    {
      cloudsEngine->setArtPressureRadius(artPressureRadius);
    }

    int artPressureExp = (int)cloudsEngine->getArtPressureExp();
    if (ImGui::SliderInt("Exponent", &artPressureExp, 1, 6))
    {
      cloudsEngine->setArtPressureExp((size_t)artPressureExp);
    }
  }

  bool isVorticityConfinementEnabled = cloudsEngine->isVorticityConfinementEnabled();
  if (ImGui::Checkbox("Enable Vorticity Confinement", &isVorticityConfinementEnabled))
  {
    cloudsEngine->enableVorticityConfinement(isVorticityConfinementEnabled);
  }
  if (isVorticityConfinementEnabled)
  {
    float vorticityConfinementCoeff = cloudsEngine->getVorticityConfinementCoeff();
    if (ImGui::SliderFloat("Vorticity Coefficient", &vorticityConfinementCoeff, 0.0f, 0.001f, "%.4f"))
    {
      cloudsEngine->setVorticityConfinementCoeff(vorticityConfinementCoeff);
    }

    float xsphViscosityCoeff = cloudsEngine->getXsphViscosityCoeff();
    if (ImGui::SliderFloat("Viscosity Coefficient", &xsphViscosityCoeff, 0.0f, 0.001f, "%.4f"))
    {
      cloudsEngine->setXsphViscosityCoeff(xsphViscosityCoeff);
    }
  }

  ImGui::Spacing();
  ImGui::Text("Clouds Parameters");
  ImGui::Spacing();

  bool isTempSmoothingEnabled = cloudsEngine->isTempSmoothingEnabled();
  if (ImGui::Checkbox("Enable Temperature Smoothing", &isTempSmoothingEnabled))
  {
    cloudsEngine->enableTempSmoothing(isTempSmoothingEnabled);
  }

  float groundHeatCoeff = cloudsEngine->getGroundHeatCoeff();
  if (ImGui::SliderFloat("Ground Heat Coefficient", &groundHeatCoeff, 0.0f, 1000.0f, "%.4f"))
  {
    cloudsEngine->setGroundHeatCoeff(groundHeatCoeff);
  }

  float buoyancyCoeff = cloudsEngine->getBuoyancyCoeff();
  if (ImGui::SliderFloat("Buoyancy Coefficient", &buoyancyCoeff, 0.0f, 5.0f, "%.4f"))
  {
    cloudsEngine->setBuoyancyCoeff(buoyancyCoeff);
  }

  float gravCoeff = cloudsEngine->getGravCoeff();
  if (ImGui::SliderFloat("Gravity Coefficient", &gravCoeff, 0.0f, 0.1f, "%.4f"))
  {
    cloudsEngine->setGravCoeff(gravCoeff);
  }

  float adiabaticLapseRate = cloudsEngine->getAdiabaticLapseRate();
  if (ImGui::SliderFloat("Adiabatic Lapse Rate", &adiabaticLapseRate, 0.0f, 20.0f, "%.3f"))
  {
    cloudsEngine->setAdiabaticLapseRate(adiabaticLapseRate);
  }

  float phaseTransitionRate = cloudsEngine->getPhaseTransitionRate();
  if (ImGui::SliderFloat("Phase Transition Rate", &phaseTransitionRate, 0.0f, 20.0f, "%.4f"))
  {
    cloudsEngine->setPhaseTransitionRate(phaseTransitionRate);
  }

  float latentHeatCoeff = cloudsEngine->getLatentHeatCoeff();
  if (ImGui::SliderFloat("Latent Heat Coefficient", &latentHeatCoeff, 0.0f, 0.100f, "%.4f"))
  {
    cloudsEngine->setLatentHeatCoeff(latentHeatCoeff);
  }

  float windCoeff = cloudsEngine->getWindCoeff();
  if (ImGui::SliderFloat("Wind Coefficient", &windCoeff, 0.0f, 1.0f, "%.4f"))
  {
    cloudsEngine->setWindCoeff(windCoeff);
  }

  ImGui::End();
}

void displayFluidsParameters(Physics::Fluids* fluidsEngine)
{
  if (!fluidsEngine)
    return;

  ImGui::Begin("Fluids Widget", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PushItemWidth(150);

  // Selection of the initial setup
  auto caseType = fluidsEngine->getInitialCase();

  const auto& selCaseName = (Physics::Fluids::ALL_CASES.find(caseType) != Physics::Fluids::ALL_CASES.end())
      ? Physics::Fluids::ALL_CASES.find(caseType)->second
      : Physics::Fluids::ALL_CASES.cbegin()->second;

  if (ImGui::BeginCombo("Study case", selCaseName.c_str()))
  {
    for (const auto& caseT : Physics::Fluids::ALL_CASES)
    {
      if (ImGui::Selectable(caseT.second.c_str(), caseType == caseT.first))
      {
        caseType = caseT.first;

        fluidsEngine->setInitialCase(caseType);
        fluidsEngine->reset();

        LOG_DEBUG("Fluids initial case correctly switched to {}", Physics::Fluids::ALL_CASES.find(caseType)->second);
      }
    }
    ImGui::EndCombo();
  }

  json jsBlock = fluidsEngine->GetJsonBlock(0);

  //for (auto& el : jsBlock.items())
  //{
  //std::cout << el.key() << " " << el.value() << std::endl;
  //}

  ImGui::Value("Particles", (int)fluidsEngine->nbParticles());

  ImGui::Spacing();
  ImGui::Text("Fluid parameters");
  ImGui::Spacing();

  float restDensity = jsBlock["restDensity"]; //->getRestDensity();
  if (ImGui::SliderFloat("Rest Density", &restDensity, 10.0f, 1000.0f))
  {
    jsBlock["restDensity"] = restDensity;
  }

  fluidsEngine->SetJsonBlock(0, jsBlock);

  float relaxCFM = fluidsEngine->getRelaxCFM();
  if (ImGui::SliderFloat("Relax CFM", &relaxCFM, 100.0f, 1000.f))
  {
    fluidsEngine->setRelaxCFM(relaxCFM);
  }

  float timeStep = fluidsEngine->getTimeStep();
  if (ImGui::SliderFloat("Time Step", &timeStep, 0.0001f, 0.020f))
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

void displayBoidsParameters(Physics::Boids* boidsEngine)
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
  const auto nbParticles = (Utils::NbParticles)boidsEngine->nbParticles();

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
*/

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
  if (ImGui::SliderFloat(name.c_str(), &floatVal, minVal, maxVal))
  {
    js.at(0) = floatVal;
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

  ImGui::Value("Particles", (int)physicsEngine->nbParticles());

  // Retrieve input json from the physics engine with all available parameters
  json js = physicsEngine->getInputJson();
  // Draw all items from input json
  drawImguiObjectFromJson(js);
  // Update physics engine with new parameters values if any
  physicsEngine->updateInputJson(js);

  /*
  auto* boidsEngine = dynamic_cast<Physics::Boids*>(physicsEngine.get());
  auto* fluidsEngine = dynamic_cast<Physics::Fluids*>(physicsEngine.get());
  auto* cloudsEngine = dynamic_cast<Physics::Clouds*>(physicsEngine.get());

  // First default pos
  ImGui::SetNextWindowPos(ImVec2(15, 355), ImGuiCond_FirstUseEver);

  if (boidsEngine)
    displayBoidsParameters(boidsEngine);
  else if (fluidsEngine)
    displayFluidsParameters(fluidsEngine);
  else if (cloudsEngine)
    displayCloudsParameters(cloudsEngine);
*/
}