#include "Model.hpp"

#include "Boids.hpp"
#include "Clouds.hpp"
#include "Fluids.hpp"

#include "Logging.hpp"

#include "ocl/Context.hpp"

std::unique_ptr<Physics::Model> Physics::CreateModel(Physics::ModelType type, Physics::ModelParams params)
{
  switch ((int)type)
  {
  case Physics::ModelType::BOIDS:
    return std::make_unique<Physics::Boids>(params);
  case Physics::ModelType::FLUIDS:
    return std::make_unique<Physics::Fluids>(params);
  default:
    return nullptr;
  }
  return nullptr;
}

Physics::Model::~Model()
{
  // We don't want any CL presence on header side, as it is shared with UI
  CL::Context::Get().release();
}

bool Physics::Model::isProfilingEnabled() const
{
  CL::Context& clContext = Physics::CL::Context::Get();
  return clContext.isProfiling();
}

void Physics::Model::enableProfiling(bool enable)
{
  CL::Context& clContext = Physics::CL::Context::Get();
  clContext.enableProfiler(enable);
}

bool Physics::Model::isUsingIGPU() const
{
  const std::string& platformName = Physics::CL::Context::Get().getPlatformName();

  return (platformName.find("Intel") != std::string::npos);
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