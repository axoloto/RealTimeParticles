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
  OclModel(ModelParams params, KernelInputs... kernelInputs)
      : Model(params)
  {
    // Adding all inputs
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

  // json GetJsonBlock(int i) const { return i < m_jsonBlocks.size() ? m_jsonBlocks.at(i) : json {}; }
  // void SetJsonBlock(int i, const json& js)
  // {
  //    if (i < m_jsonBlocks.size())
  //      m_jsonBlocks[i] = js;
  //  }

  protected:
  std::vector<std::variant<KernelInputs...>> m_kernelInputs;
  // std::vector<json> m_jsonBlocks;
};
}
