#pragma once

#include "OclModel.hpp"

#include "../utils/RadixSort.hpp"
#include "../utils/Target.hpp"

#include <array>
#include <memory>
#include <vector>

namespace Physics::CL
{
struct BoidsRuleKernelInputs
{
  cl_float velocityScale;
  cl_float alignmentScale;
  cl_float separationScale;
  cl_float cohesionScale;
};

struct TargetKernelInputs
{
  cl_float targetRadiusEffect;
  cl_int targetSignEffect;
};

class Boids : public OclModel<BoidsRuleKernelInputs, TargetKernelInputs>
{
  public:
  Boids(ModelParams params);
  ~Boids();

  void update() override;
  void reset() override;

  Math::float3 targetPos() const override
  {
    return m_target.pos();
  }
  bool isTargetActivated() const override { return m_target.isActivated(); }
  bool isTargetVisible() const override { return m_target.isVisible(); }

  private:
  void initBoidsParticles();
  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;
  void updateBoidsParamsInKernel();
  void updateGridParamsInKernel();

  void transferJsonInputsToModel() override;
  void transferKernelInputsToGPU() override;

  bool m_activeAlignment;
  bool m_activeCohesion;
  bool m_activeSeparation;

  float m_scaleAlignment;
  float m_scaleCohesion;
  float m_scaleSeparation;

  bool m_simplifiedMode;
  size_t m_maxNbPartsInCell;

  Target m_target;

  RadixSort m_radixSort;
};
}