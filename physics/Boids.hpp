#pragma once

//#include "CL/cl.h"
#include <array>
#include <vector>

#include "Physics.hpp"
#include "ocl/Context.hpp"

namespace Core
{
class Boids : public Physics
{
  public:
  Boids(size_t numEntities, size_t gridRes,
      unsigned int pointCloudCoordVBO,
      unsigned int pointCloudColorVBO,
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

  void activateTarget(bool targets)
  {
    m_activeTargets = targets;
    updateBoidsParamsInKernel();
  }
  bool isTargetActivated() const { return m_activeTargets; }

  private:
  bool createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO, unsigned int gridDetectorVBO);
  bool createKernels();
  void updateBoidsParamsInKernel();
  void updateGridParamsInKernel();

  bool m_activeAlignment;
  bool m_activeCohesion;
  bool m_activeSeparation;

  float m_scaleAlignment;
  float m_scaleCohesion;
  float m_scaleSeparation;

  bool m_activeTargets;

  Math::float3 m_target;
  struct boidsParams
  {
    cl_float dims;
    cl_float velocity;
    cl_float scaleCohesion;
    cl_float scaleAlignment;
    cl_float scaleSeparation;
    cl_int activeTarget;
  };

  boidsParams m_boidsParams;
  cl_mem cl_boidsParamsBuff;

  struct gridParams
  {
    cl_uint gridRes;
    cl_uint numCells;
  };

  gridParams m_gridParams;
  cl_mem cl_gridParamsBuff;

  CL::Context m_clContext;
};
}