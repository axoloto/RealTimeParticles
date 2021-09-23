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

struct FluidKernelInputs
{
  cl_float effectRadius;
  cl_float restDensity;
  cl_float relaxCFM;
  cl_float timeStep;
  cl_uint dim;
};

class Fluids : public Model
{
  public:
  Fluids(ModelParams params);
  ~Fluids() = default;

  void update() override;
  void reset() override;

  //
  void setVelocity(float velocity) override
  {
    m_velocity = velocity;
    updateFluidsParamsInKernel();
  }

  private:
  //void print(const std::string& name, int nbItems);

  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;

  void updateFluidsParamsInKernel();

  bool m_simplifiedMode;
  size_t m_maxNbPartsInCell;

  std::unique_ptr<Target> m_target;

  RadixSort m_radixSort;

  FluidKernelInputs m_kernelInputs;
};
}