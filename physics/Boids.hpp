#pragma once

#include <array>
#include <vector>

#include "Model.hpp"
#include "ocl/Context.hpp"
#include "utils/RadixSort.hpp"
#include "utils/Target.hpp"

namespace Physics
{
class Boids : public Model
{
  public:
  Boids(ModelParams params);
  ~Boids() = default;

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
    return m_target ? m_target->pos() : Math::float3({ 0.0f, 0.0f, 0.0f });
  }

  void activateTarget(bool isActive)
  {
    if (m_target)
      m_target->activate(isActive);

    updateBoidsParamsInKernel();
  }
  bool isTargetActivated() const override { return m_target ? m_target->isActivated() : false; }

  void setTargetVisibility(bool isVisible)
  {
    if (m_target)
      m_target->show(isVisible);
  }
  bool isTargetVisible() const override { return m_target ? m_target->isVisible() : false; }

  void setTargetRadiusEffect(float radiusEffect)
  {
    if (m_target)
      m_target->setRadiusEffect(radiusEffect);

    updateBoidsParamsInKernel();
  }
  float targetRadiusEffect() const { return m_target ? m_target->radiusEffect() : 0.0f; }

  void setTargetSignEffect(int signEffect)
  {
    if (m_target)
      m_target->setSignEffect(signEffect);

    updateBoidsParamsInKernel();
  }
  int targetSignEffect() const { return m_target ? m_target->signEffect() : 0; }

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

  std::unique_ptr<Target> m_target;

  RadixSort m_radixSort;
};
}