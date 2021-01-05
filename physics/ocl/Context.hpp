#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include "CL/cl2.hpp"

#include <map>
#include <string>
#include <vector>

namespace Core
{
namespace CL
{
class Context
{
  public:
  Context(bool profilingEnabled = true);
  ~Context() = default;

  bool init();

  bool createProgram(std::string programName, std::string sourcePath, std::string specificBuildOptions);
  bool createGLBuffer(std::string GLBufferName, unsigned int VBOIndex, cl_mem_flags memoryFlags);
  bool createBuffer(std::string bufferName, size_t bufferSize, cl_mem_flags memoryFlags);

  bool createKernel(std::string programName, std::string kernelName, std::vector<std::string> argNames);
  bool setKernelArg(std::string kernelName, cl_uint argIndex, size_t argSize, const void* value);
  bool runKernel(std::string kernelName, size_t numWorkItems);

  bool acquireGLBuffers(const std::vector<std::string>& GLBufferNames) { return interactWithGLBuffers(GLBufferNames, interOpCLGL::ACQUIRE); }
  bool releaseGLBuffers(const std::vector<std::string>& GLBufferNames) { return interactWithGLBuffers(GLBufferNames, interOpCLGL::RELEASE); }

  bool mapAndSendBufferToDevice(std::string bufferName, const void* bufferPtr, size_t bufferSize);

  private:
  bool findPlatforms();
  bool findGPUDevices();
  bool createContext();
  bool createCommandQueue();

  enum class interOpCLGL
  {
    ACQUIRE,
    RELEASE
  };
  bool interactWithGLBuffers(const std::vector<std::string>& GLBufferNames, interOpCLGL interaction);

  cl::Platform cl_platform;
  cl::Device cl_device;
  cl::Context cl_context;
  cl::CommandQueue cl_queue;

  std::map<std::string, cl::Program> m_programsMap;
  std::map<std::string, cl::Kernel> m_kernelsMap;
  std::map<std::string, cl::Buffer> m_buffersMap;
  std::map<std::string, cl::BufferGL> m_GLBuffersMap;

  bool m_isKernelProfilingEnabled;

  bool m_init;

  std::vector<cl::Platform> m_allPlatforms;
  std::vector<std::pair<cl::Platform, std::vector<cl::Device>>> m_allGPUsWithInteropCLGL;
};
} //CL
} //Core