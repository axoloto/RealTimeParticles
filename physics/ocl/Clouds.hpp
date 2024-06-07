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
  //
  cl_float relaxCFM = 600.0f;
  //
  cl_float initVaporDensityCoeff = 0.75f;
  //
  cl_float windCoeff = 1.0f;
};

class Clouds : public OclModel<FluidKernelInputs, CloudKernelInputs>
{
  public:
  // List of implemented cases
  enum CaseType
  {
    CUMULUS = 0,
    HOMOGENEOUS = 1
  };

  struct CompareCaseType
  {
    bool operator()(const CaseType& caseA, const CaseType& caseB) const
    {
      return (int)caseA < (int)caseB;
    }
  };
  // Static member vars must be initialized outside of the class in the global scope
  static const std::map<CaseType, std::string, CompareCaseType> ALL_CASES;

  Clouds(ModelParams params);
  ~Clouds();

  void update() override;
  void reset() override;

  void setInitialCase(CaseType caseT) { m_initialCase = caseT; }
  const CaseType getInitialCase() const { return m_initialCase; }

  //
  void setRestDensity(float restDensity);
  float getRestDensity() const;
  //
  void setRelaxCFM(float relaxCFM);
  float getRelaxCFM() const;
  //
  void setTimeStep(float timeStep);
  float getTimeStep() const;
  //
  void setNbJacobiIters(size_t nbIters);
  size_t getNbJacobiIters() const;
  //
  void enableArtPressure(bool enable);
  bool isArtPressureEnabled() const;
  //
  void setArtPressureRadius(float radius);
  float getArtPressureRadius() const;
  //
  void setArtPressureExp(size_t exp);
  size_t getArtPressureExp() const;
  //
  void setArtPressureCoeff(float coeff);
  float getArtPressureCoeff() const;
  //
  void enableVorticityConfinement(bool enable);
  bool isVorticityConfinementEnabled() const;
  //
  void setVorticityConfinementCoeff(float coeff);
  float getVorticityConfinementCoeff() const;
  //
  void setXsphViscosityCoeff(float coeff);
  float getXsphViscosityCoeff() const;
  //
  void setGroundHeatCoeff(float coeff);
  float getGroundHeatCoeff() const;
  //
  void setBuoyancyCoeff(float coeff);
  float getBuoyancyCoeff() const;
  //
  void setAdiabaticLapseRate(float rate);
  float getAdiabaticLapseRate() const;
  //
  void setPhaseTransitionRate(float rate);
  float getPhaseTransitionRate() const;
  //
  void setLatentHeatCoeff(float coeff);
  float getLatentHeatCoeff() const;
  //
  void setGravCoeff(float coeff);
  float getGravCoeff() const;
  //
  void setWindCoeff(float coeff);
  float getWindCoeff() const;
  //
  void enableTempSmoothing(bool enable);
  bool isTempSmoothingEnabled() const;

  private:
  bool createProgram() const;
  bool createBuffers();
  bool createKernels() const;

  void initCloudsParticles();

  void updateFluidsParamsInKernels();
  void updateCloudsParamsInKernels();

  bool m_simplifiedMode;

  size_t m_maxNbPartsInCell;

  size_t m_nbJacobiIters;

  RadixSort m_radixSort;

  std::unique_ptr<FluidKernelInputs> m_fluidKernelInputs;
  std::unique_ptr<CloudKernelInputs> m_cloudKernelInputs;

  CaseType m_initialCase;
};
}