#pragma once

#include "Model.hpp"
#include "utils/RadixSort.hpp"

#include <array>
#include <memory>
#include <vector>

namespace Physics
{
// Forward decl
struct FluidKernelInputs;
struct CloudKernelInputs;

class Clouds : public Model
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