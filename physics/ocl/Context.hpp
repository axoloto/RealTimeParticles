#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

//#include "CL/cl.h"
//#include "CL/cl_gl.h" // WIP

#include "CL/cl2.hpp"

//#include "CL/opencl.hpp"
//#include <CL/opencl.hpp>
#include <array>
//#include <khronos-opencl-clhpp/opencl.hpp>
#include <string>
#include <vector>

namespace Core
{
namespace CL
{
class Context
{
  public:
  Context(std::string sourcePath, std::string specificBuildOptions);
  ~Context() = default;

  bool init();

  // bool createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
  // bool acquireGLBuffers(const std::vector<cl_mem>& GLBuffers);
  // bool releaseGLBuffers(const std::vector<cl_mem>& GLBuffers);
  // bool createKernels();
  // void runKernel(cl_kernel kernel, double* profilingTimeMs = nullptr);
  // void updateBoidsParamsInKernel();

  cl::Platform cl_Platform;
  cl::Device cl_Device;
  cl::Context cl_Context;
  cl::Program cl_Program;
  cl::CommandQueue cl_Queue;

  cl_platform_id cl_platform;
  cl_device_id cl_device;
  cl_context cl_context;
  cl_program cl_program;
  cl_command_queue cl_queue;

  bool m_kernelProfilingEnabled;

  private:
  bool findPlatform();
  bool findDevice();
  bool createContext();
  bool createAndBuildProgram();
  bool createCommandQueue();

  std::string m_preferredPlatformName;
  std::string m_sourceFilePath;
  std::string m_specificBuildOptions;
};
} //CL
} //Core