#pragma once

#include "OclModel.hpp"

#include "../utils/RadixSort.hpp"
#include "Parameters.hpp"

#include <array>
#include <memory>
#include <vector>

// Position based fluids model based on NVIDIA paper
// Macklin and Muller 2013. "Position Based Fluids"

namespace Physics::CL
{
struct FluidKernelInputs
{
  cl_float restDensity = 450.0f;
  cl_float relaxCFM = 600.0f;
  cl_float timeStep = 0.010f;
  cl_uint dim = 3;
  // Artifical pressure if enabled will try to reduce tensile instability
  cl_uint isArtPressureEnabled = 1;
  cl_float artPressureRadius = 0.006f;
  cl_float artPressureCoeff = 0.001f;
  cl_uint artPressureExp = 4;
  // Vorticity confinement if enabled will try to replace lost energy due to virtual damping
  cl_uint isVorticityConfEnabled = 1;
  cl_float vorticityConfCoeff = 0.0004f;
  cl_float xsphViscosityCoeff = 0.0001f;
};

class Fluids : public OclModel<FluidKernelInputs>
{
  public:
  Fluids(ModelParams params);
  ~Fluids();

  void update() override;
  void reset() override;

  // OclModel.hpp
  void transferJsonInputsToModel() override;
  void transferKernelInputsToGPU() override;

  private:
  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;

  void initFluidsParticles();
  void updateFluidsParamsInKernels();

  bool m_simplifiedMode;

  size_t m_maxNbPartsInCell;

  // Input available in UI through input json
  size_t m_nbJacobiIters;

  RadixSort m_radixSort;
};
}