#pragma once

#include "CL/cl.h"
#include <array>
#include <vector>

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

  void setScaleAlignment(float alignment) { m_scaleAlignment = alignment; }
  float getScaleAlignment() const { return m_scaleAlignment; }

  void setScaleCohesion(float cohesion) { m_scaleCohesion = cohesion; }
  float getScaleCohesion() const { return m_scaleCohesion; }

  void setScaleSeparation(float separation) { m_scaleSeparation = separation; }
  float getScaleSeparation() const { return m_scaleSeparation; }

  void setActivateTargets(bool targets) { m_activeTargets = targets; }
  bool getActivateTargets() { return m_activeTargets; }

  void setActivateAlignment(bool alignment) { m_activeAlignment = alignment; }
  bool getActivateAlignment() const { return m_activeAlignment; }

  void setActivateCohesion(bool cohesion) { m_activeCohesion = cohesion; }
  bool getActivateCohesion() const { return m_activeCohesion; }

  void setActivateSeparation(bool separation) { m_activeSeparation = separation; }
  bool getActivateSeparation() const { return m_activeSeparation; }

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

  cl_platform_id cl_platform;
  cl_device_id cl_device;
  cl_context cl_context;
  cl_program cl_program;
  cl_command_queue cl_queue;

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

  bool m_kernelProfilingEnabled;
};
}