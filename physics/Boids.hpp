#pragma once

#include <array>
#include <chrono>
#include <vector>

#include "Physics.hpp"
#include "ocl/Context.hpp"
#include "utils/RadixSort.hpp"

namespace Core
{
using clock = std::chrono::high_resolution_clock;
class Boids : public Physics
{
  public:
  Boids(size_t numEntities, size_t boxSize, size_t gridRes, float velocity,
      unsigned int pointCloudCoordVBO,
      unsigned int cameraCoordVBO,
      unsigned int gridDetectorVBO);
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

  void activateTarget(bool target)
  {
    m_activeTargets = target;
    updateBoidsParamsInKernel();
  }
  bool isTargetActivated() const { return m_activeTargets; }

  void setTargetRadiusEffect(float radiusEffect)
  {
    m_targetRadiusEffect = radiusEffect;
    updateBoidsParamsInKernel();
  }
  float targetRadiusEffect() const { return m_targetRadiusEffect; }

  void setTargetSignEffect(int signEffect)
  {
    m_targetSign = signEffect;
    updateBoidsParamsInKernel();
  }
  int targetSignEffect() const { return m_targetSign; }

  private:
  bool createProgram() const;
  bool createBuffers(unsigned int pointCloudCoordVBO, unsigned int cameraCoordVBO, unsigned int gridDetectorVBO) const;
  bool createKernels() const;
  void updateBoidsParamsInKernel();
  void updateGridParamsInKernel();

  bool m_activeAlignment;
  bool m_activeCohesion;
  bool m_activeSeparation;

  float m_scaleAlignment;
  float m_scaleCohesion;
  float m_scaleSeparation;

  bool m_activeTargets;

  size_t m_maxNbPartsInCell;

  Math::float3 m_target;
  float m_targetRadiusEffect;
  int m_targetSign;

  RadixSort m_radixSort;

  std::chrono::steady_clock::time_point m_time;
};
}