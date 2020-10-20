#pragma once

#include "CL/cl.h"
#include <array>
#include <vector>

namespace Core
{
namespace CL
{
class Context
{
  public:
  Context() {};
  ~Context();

  //private:
  bool init();
  bool isExtensionSupported(cl_device_id device, const char* extension);

  // bool createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
  // bool acquireGLBuffers(const std::vector<cl_mem>& GLBuffers);
  // bool releaseGLBuffers(const std::vector<cl_mem>& GLBuffers);
  // bool createKernels();
  // void runKernel(cl_kernel kernel, double* profilingTimeMs = nullptr);
  // void updateBoidsParamsInKernel();

  cl_platform_id cl_platform;
  cl_device_id cl_device;
  cl_context cl_context;
  cl_program cl_program;
  cl_command_queue cl_queue;

  bool m_kernelProfilingEnabled;

  std::string m_preferredPlatformName { "NVIDIA" };
  cl_device_type m_preferredDeviceType { CL_DEVICE_TYPE_GPU };
};
} //CL
} //Core