#pragma once

#include <array>
#include <chrono>
#include <vector>

#include "Model.hpp"
#include "ocl/Context.hpp"
#include "utils/RadixSort.hpp"
#include "utils/Target.hpp"

// Position based fluids model based on NVIDIA paper
// Muller and al. 2013. "Position Based Fluids"

namespace Physics
{
using clock = std::chrono::high_resolution_clock;

// List of implemented cases
enum CaseType
{
  DAM = 0,
  DROP = 1
};

struct CompareCaseType
{
  bool operator()(const CaseType& caseA, const CaseType& caseB) const
  {
    return (int)caseA < (int)caseB;
  }
};

static const std::map<CaseType, std::string, CompareCaseType> ALL_CASES {
  { CaseType::DAM, "Dam" },
  { CaseType::DROP, "Drop" },
};

struct FluidKernelInputs
{
  cl_float effectRadius = 0.3f;
  cl_float restDensity = 450.0f;
  cl_float relaxCFM = 600.0f;
  cl_float timeStep = 0.008f;
  cl_uint dim = 3;
};

class Fluids : public Model
{
  public:
  Fluids(ModelParams params);
  ~Fluids() = default;

  void update() override;
  void reset() override;

  //
  void setEffectRadius(float effectRadius)
  {
    m_kernelInputs.effectRadius = (cl_float)effectRadius;
    updateFluidsParamsInKernel();
  }
  float getEffectRadius() const { return (float)m_kernelInputs.effectRadius; }

  //
  void setRestDensity(float restDensity)
  {
    m_kernelInputs.restDensity = (cl_float)restDensity;
    updateFluidsParamsInKernel();
  }
  float getRestDensity() const { return (float)m_kernelInputs.restDensity; }

  //
  void setRelaxCFM(float relaxCFM)
  {
    m_kernelInputs.relaxCFM = (cl_float)relaxCFM;
    updateFluidsParamsInKernel();
  }
  float getRelaxCFM() const { return (float)m_kernelInputs.relaxCFM; }

  //
  void setTimeStep(float timeStep)
  {
    m_kernelInputs.timeStep = (cl_float)timeStep;
    updateFluidsParamsInKernel();
  }
  float getTimeStep() const { return (float)m_kernelInputs.timeStep; }

  private:
  //void print(const std::string& name, int nbItems);

  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;

  void initFluid() const;
  void updateFluidsParamsInKernel();

  bool m_simplifiedMode;
  size_t m_maxNbPartsInCell;

  std::unique_ptr<Target> m_target;

  RadixSort m_radixSort;

  FluidKernelInputs m_kernelInputs;

  CaseType m_initialCase;
};
}