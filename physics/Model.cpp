#include "Model.hpp"

#include "ocl/Context.hpp"

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

