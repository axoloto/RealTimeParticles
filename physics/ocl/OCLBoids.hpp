#pragma once

#include "Boids.hpp"
#include "CL/cl.h"
#include <array>
#include <vector>

namespace Core
{
class OCLBoids : public Boids
{
  public:
  OCLBoids(int boxSize, int numEntities, unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
  ~OCLBoids();

  void updatePhysics() override;
  void resetParticles() override;

  private:
  bool initOpenCL();
  bool createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
  bool acquireGLBuffers(const std::vector<cl_mem>& GLBuffers);
  bool releaseGLBuffers(const std::vector<cl_mem>& GLBuffers);
  bool createKernels();
  void runKernel(cl_kernel kernel, double* profilingTimeMs = nullptr);

  bool m_init;

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
  cl_kernel cl_updatePosKernel;

  bool m_kernelProfilingEnabled;
};
}