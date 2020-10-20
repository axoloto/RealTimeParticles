#pragma once

#include "CL/cl.h"
#include <array>
#include <vector>

#include "Context.hpp"
#include "Physics.hpp"

namespace Core
{
class OCLBoids : public Physics
{
  public:
  OCLBoids(int numEntities, unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
  ~OCLBoids();

  void update() override;
  void reset() override;

  //

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
  bool isTargetActivated() { return m_activeTargets; }

  private:
  bool initOpenCL();
  bool createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
  bool acquireGLBuffers(const std::vector<cl_mem>& GLBuffers);
  bool releaseGLBuffers(const std::vector<cl_mem>& GLBuffers);
  bool createKernels();
  void runKernel(cl_kernel kernel, double* profilingTimeMs = nullptr);
  void updateBoidsParamsInKernel();

  bool m_activeAlignment;
  bool m_activeCohesion;
  bool m_activeSeparation;

  float m_scaleAlignment;
  float m_scaleCohesion;
  float m_scaleSeparation;

  bool m_activeTargets;

  Math::float3 m_target;

  cl_mem cl_colorBuff;
  cl_mem cl_posBuff;
  cl_mem cl_accBuff;
  cl_mem cl_velBuff;

  cl_kernel cl_colorKernel;
  cl_kernel cl_initPosKernel;
  cl_kernel cl_boidsRulesKernel;
  cl_kernel cl_updateVelKernel;
  cl_kernel cl_updatePosCyclicWallsKernel;
  cl_kernel cl_updatePosBouncingWallsKernel;

  struct boidsParams
  {
    cl_float velocity;
    cl_float scaleCohesion;
    cl_float scaleAlignment;
    cl_float scaleSeparation;
    cl_int activeTarget;
  };

  boidsParams m_boidsParams;
  cl_mem cl_boidsParamsBuff;

  std::unique_ptr<CL::Context> m_clContext;
};
}