#pragma once

#include "OclModel.hpp"

#include "../utils/RadixSort.hpp"

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

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(FluidKernelInputs, restDensity, relaxCFM, timeStep,
      dim, isArtPressureEnabled, artPressureRadius, artPressureCoeff, artPressureExp, isVorticityConfEnabled, vorticityConfCoeff, xsphViscosityCoeff);
};

class Fluids : public OclModel<FluidKernelInputs>
{
  public:
  // List of implemented cases
  enum CaseType
  {
    DAM = 0,
    BOMB = 1,
    DROP = 2
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

  Fluids(ModelParams params);
  ~Fluids();

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

  private:
  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;

  void initFluidsParticles();
  void updateFluidsParamsInKernels();

  bool m_simplifiedMode;

  size_t m_maxNbPartsInCell;

  size_t m_nbJacobiIters;

  RadixSort m_radixSort;

  FluidKernelInputs m_kernelInputs;

  CaseType m_initialCase;
};
}