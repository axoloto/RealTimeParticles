#pragma once

#include "../Model.hpp"
#include "Context.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <variant>
//class Model;

namespace Physics::CL
{
template <typename... KernelInputs>
class OclModel : public Model
{
  public:
  OclModel(ModelParams params, KernelInputs... kernelInputs, json inputJson = {})
      : Model(params, inputJson)
  {
    // Adding all inputs to kernel inputs for GPU-CPU interaction
    (m_kernelInputs.push_back(kernelInputs), ...);
  };

  ~OclModel()
  {
    CL::Context::Get().release();
  }

  bool isProfilingEnabled() const override
  {
    CL::Context& clContext = Physics::CL::Context::Get();
    return clContext.isProfiling();
  }

  void enableProfiling(bool enable) override
  {
    CL::Context& clContext = Physics::CL::Context::Get();
    clContext.enableProfiler(enable);
  }

  bool isUsingIGPU() const override
  {
    const std::string& platformName = Physics::CL::Context::Get().getPlatformName();

    return (platformName.find("Intel") != std::string::npos);
  }

  virtual void transferKernelInputsToGPU() {}; // = 0

  template <typename T>
  T& GetKernelInput(int index)
  {
    // risky business
    return std::get<T>(m_kernelInputs.at(index));
  }

  protected:
  std::vector<std::variant<KernelInputs...>> m_kernelInputs;
};
}
