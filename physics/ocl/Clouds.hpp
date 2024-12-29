#pragma once

#include "Fluids.hpp"
#include "OclModel.hpp"

#include "../utils/RadixSort.hpp"

#include <array>
#include <memory>
#include <vector>

namespace Physics::CL
{
// Clouds params for clouds-specific physics
struct CloudKernelInputs
{
  cl_uint dim = 3;
  // Must be always equal to timeStep of FluidKernelInputs
  cl_float timeStep = 0.01f;
  // Must be equal to rest density of FluidKernelInputs
  cl_float restDensity = 400.0f;
  // groundHeatCoeff * timeStep = temperature increase due to ground being a heat source
  // Effect is limited to closest part of the atmosphere and then exponentially reduced to null value
  cl_float groundHeatCoeff = 10.0f; // i.e here 0.4K/iteration, at 30fps -> 12K/s increase
  // Buoyancy makes warmer particles to go up and colder ones to go down
  cl_float buoyancyCoeff = 0.10f;
  cl_float gravCoeff = 0.0005f;
  // Adiabatic cooling makes the air parcels to cool down when going up
  cl_float adiabaticLapseRate = 5.0f;
  // Phase transition rate decides how fast waper transitions
  // between vapor and liquid (clouds = droplets)
  cl_float phaseTransitionRate = 0.3485f;
  // When particles transition from vapor to liquid, they released heat
  // increasing their temperature, making them going up some more due to buoyancy
  cl_float latentHeatCoeff = 0.07f;
  // Enable constraint on temperature field, forcing its Laplacian field to be null
  // It helps uniformizing the temperature across particles
  cl_uint isTempSmoothingEnabled = 1;
  // Must be equal to relaxCFM value of FluidKernelInputs
  cl_float relaxCFM = 600.0f;
  //
  cl_float initVaporDensityCoeff = 0.75f;
  //
  cl_float windCoeff = 1.0f;
};

class Clouds : public OclModel<FluidKernelInputs, CloudKernelInputs>
{
  public:
  Clouds(ModelParams params);
  ~Clouds();

  void update() override;
  void reset() override;

  private:
  bool createProgram() const;
  bool createBuffers();
  bool createKernels() const;

  void initCloudsParticles();

  void updateFluidsParamsInKernels();
  void updateCloudsParamsInKernels();

  void transferJsonInputsToModel(json& inputJson) override;
  void transferKernelInputsToGPU() override;

  bool m_simplifiedMode;

  size_t m_maxNbPartsInCell;

  size_t m_nbJacobiIters;

  RadixSort m_radixSort;

  // To simplify access to the different kernel inputs that are stored at OclModel level
  FluidKernelInputs* m_fluidKernelInputs;
  CloudKernelInputs* m_cloudKernelInputs;
};
}