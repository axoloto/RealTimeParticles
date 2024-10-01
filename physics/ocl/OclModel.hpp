#pragma once

#include "../Model.hpp"
#include "Context.hpp"

#include <variant>

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

  // Model.hpp
  void updateModelWithInputJson() override
  {
    // First transfer inputs from json to model and kernel inputs
    transferJsonInputsToModel();
    // Then transfer kernel inputs from CPU to GPU
    transferKernelInputsToGPU();
  }

  virtual void transferJsonInputsToModel() = 0;
  virtual void transferKernelInputsToGPU() = 0;

  template <typename T>
  T& getKernelInput(int index)
  {
    // risky business
    return std::get<T>(m_kernelInputs.at(index));
  }

  size_t getNbKernelInputs() { return m_kernelInputs.size(); }

  protected:
  std::vector<std::variant<KernelInputs...>> m_kernelInputs;
};
}
