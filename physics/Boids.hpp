#pragma once

#include "Model.hpp"
#include "utils/Target.hpp"
#include "utils/RadixSort.hpp"

#include <array>
#include <vector>
#include <memory>

namespace Physics
{
class Boids : public Model
{
  public:
  Boids(ModelParams params);
  ~Boids();

  void update() override;
  void reset() override;

  //
  void setVelocity(float velocity) override
  {
    m_velocity = velocity;
    updateBoidsParamsInKernel();
  }

  void setScaleAlignment(float alignment)
  {
    m_scaleAlignment = alignment;
    updateBoidsParamsInKernel();
  }
  float scaleAlignment() const { return m_scaleAlignment; }

  void activateAlignment(bool alignment)
  {
    m_activeAlignment = alignment;
    updateBoidsParamsInKernel();
  }
  bool isAlignmentActivated() const { return m_activeAlignment; }

  //

  void setScaleCohesion(float cohesion)
  {
    m_scaleCohesion = cohesion;
    updateBoidsParamsInKernel();
  }
  float scaleCohesion() const { return m_scaleCohesion; }

  void activateCohesion(bool cohesion)
  {
    m_activeCohesion = cohesion;
    updateBoidsParamsInKernel();
  }
  bool isCohesionActivated() const { return m_activeCohesion; }

  //

  void setScaleSeparation(float separation)
  {
    m_scaleSeparation = separation;
    updateBoidsParamsInKernel();
  }
  float scaleSeparation() const { return m_scaleSeparation; }

  void activateSeparation(bool separation)
  {
    m_activeSeparation = separation;
    updateBoidsParamsInKernel();
  }
  bool isSeparationActivated() const { return m_activeSeparation; }

  //
  Math::float3 targetPos() const override
  {
    return m_target.pos();
  }

  void activateTarget(bool isActive)
  {
    m_target.activate(isActive);
    updateBoidsParamsInKernel();
  }
  bool isTargetActivated() const override { return m_target.isActivated(); }

  void setTargetVisibility(bool isVisible)
  {
    m_target.show(isVisible);
  }
  bool isTargetVisible() const override { return m_target.isVisible(); }

  void setTargetRadiusEffect(float radiusEffect)
  {
    m_target.setRadiusEffect(radiusEffect);
    updateBoidsParamsInKernel();
  }
  float targetRadiusEffect() const { return m_target.radiusEffect(); }

  void setTargetSignEffect(int signEffect)
  {
    m_target.setSignEffect(signEffect);
    updateBoidsParamsInKernel();
  }
  int targetSignEffect() const { return m_target.signEffect(); }

  private:
  void initBoidsParticles();
  bool createProgram() const;
  bool createBuffers() const;
  bool createKernels() const;
  void updateBoidsParamsInKernel();
  void updateGridParamsInKernel();

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