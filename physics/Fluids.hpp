#pragma once

#include <array>
#include <vector>

#include "Model.hpp"
#include "ocl/Context.hpp"
#include "utils/RadixSort.hpp"

// Position based fluids model based on NVIDIA paper
// Macklin and Muller 2013. "Position Based Fluids"

namespace Physics
{
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

static const std::map<CaseType, std::string, CompareCaseType> ALL_FLUID_CASES {
  { CaseType::DAM, "Dam" },
  { CaseType::BOMB, "Bomb" },
  { CaseType::DROP, "Drop" },
};

struct FluidKernelInputs
{
  cl_float effectRadius = 0.3f;
  cl_float restDensity = 450.0f;
  cl_float relaxCFM = 600.0f;
  cl_float timeStep = 0.008f;
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

class Fluids : public Model
{
  public:
  Fluids(ModelParams params);
  ~Fluids() = default;

  void update() override;
  void reset() override;

  void setInitialCase(CaseType caseT) { m_initialCase = caseT; }
  const CaseType getInitialCase() const { return m_initialCase; }

  // Not giving access to it for now.
  // Strongly connected to grid resolution which is not available as parameter,
  // in order to maintain cohesion between boids and fluids models
  /*
  void setEffectRadius(float effectRadius)
  {
    m_kernelInputs.effectRadius = (cl_float)effectRadius;
    updateFluidsParamsInKernel();
  }
  */
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

  //
  void setNbJacobiIters(size_t nbIters)
  {
    m_nbJacobiIters = nbIters;
  }
  size_t getNbJacobiIters() const { return m_nbJacobiIters; }

  //
  void enableArtPressure(bool enable)
  {
    m_kernelInputs.isArtPressureEnabled = (cl_uint)enable;
    updateFluidsParamsInKernel();
  }
  bool isArtPressureEnabled() const { return (bool)m_kernelInputs.isArtPressureEnabled; }

  //
  void setArtPressureRadius(float radius)
  {
    m_kernelInputs.artPressureRadius = (cl_float)radius;
    updateFluidsParamsInKernel();
  }
  float getArtPressureRadius() const { return (float)m_kernelInputs.artPressureRadius; }

  //
  void setArtPressureExp(size_t exp)
  {
    m_kernelInputs.artPressureExp = (cl_uint)exp;
    updateFluidsParamsInKernel();
  }
  size_t getArtPressureExp() const { return (size_t)m_kernelInputs.artPressureExp; }

  //
  void setArtPressureCoeff(float coeff)
  {
    m_kernelInputs.artPressureCoeff = (cl_float)coeff;
    updateFluidsParamsInKernel();
  }
  float getArtPressureCoeff() const { return (float)m_kernelInputs.artPressureCoeff; }

  //
  void enableVorticityConfinement(bool enable)
  {
    m_kernelInputs.isVorticityConfEnabled = (cl_uint)enable;
    updateFluidsParamsInKernel();
  }
  bool isVorticityConfinementEnabled() const { return (bool)m_kernelInputs.isVorticityConfEnabled; }

  //
  void setVorticityConfinementCoeff(float coeff)
  {
    m_kernelInputs.vorticityConfCoeff = (cl_float)coeff;
    updateFluidsParamsInKernel();
  }
  float getVorticityConfinementCoeff() const { return (float)m_kernelInputs.vorticityConfCoeff; }

  //
  void setXsphViscosityCoeff(float coeff)
  {
    m_kernelInputs.xsphViscosityCoeff = (cl_float)coeff;
    updateFluidsParamsInKernel();
  }
  float getXsphViscosityCoeff() const { return (float)m_kernelInputs.xsphViscosityCoeff; }

  private:
  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;

  void initFluidsParticles();
  void updateFluidsParamsInKernel();

  bool m_simplifiedMode;

  size_t m_maxNbPartsInCell;

  size_t m_nbJacobiIters;

  RadixSort m_radixSort;

  FluidKernelInputs m_kernelInputs;

  CaseType m_initialCase;
};
}