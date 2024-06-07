#include "Model.hpp"

#include "ocl/Boids.hpp"
#include "ocl/Clouds.hpp"
#include "ocl/Context.hpp"
#include "ocl/Fluids.hpp"

#include "Logging.hpp"

std::unique_ptr<Physics::Model> Physics::CreateModel(Physics::ModelType type, Physics::ModelParams params)
{
  switch ((int)type)
  {
  case Physics::ModelType::BOIDS:
    return std::make_unique<Physics::CL::Boids>(params);
  case Physics::ModelType::FLUIDS:
    return std::make_unique<Physics::CL::Fluids>(params);
  case Physics::ModelType::CLOUDS:
    return std::make_unique<Physics::CL::Clouds>(params);
  default:
    return nullptr;
  }
  return nullptr;
}

void Physics::Model::setCurrentDisplayedQuantity(const std::string& name)
{
  auto foundQuantity = m_allDisplayableQuantities.find(name);

  if (foundQuantity != m_allDisplayableQuantities.end())
  {
    m_currentDisplayedQuantityName = foundQuantity->first;
    LOG_INFO("Quantity {} selected for display", name);
  }
  else
  {
    LOG_ERROR("Quantity {} does not exist in current model", name);
  };
}