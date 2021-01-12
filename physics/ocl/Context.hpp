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
  static Context& Get();

  bool isInit() const { return m_init; }

  bool createProgram(std::string programName, std::string sourcePath, std::string specificBuildOptions);
  bool createGLBuffer(std::string GLBufferName, unsigned int VBOIndex, cl_mem_flags memoryFlags);
  bool createBuffer(std::string bufferName, size_t bufferSize, cl_mem_flags memoryFlags);
  bool loadBufferFromHost(std::string bufferName, size_t offset, size_t sizeToFill, const void* hostPtr);
  bool unloadBufferFromDevice(std::string bufferName, size_t offset, size_t sizeToFill, void* hostPtr);

  bool createKernel(std::string programName, std::string kernelName, std::vector<std::string> argNames);
  bool setKernelArg(std::string kernelName, cl_uint argIndex, size_t argSize, const void* value);
  bool setKernelArg(std::string kernelName, cl_uint argIndex, const std::string& bufferName);
  bool runKernel(std::string kernelName, size_t numFlobalWorkItems, size_t numLocalWorkItems = 0);

  bool acquireGLBuffers(const std::vector<std::string>& GLBufferNames) { return interactWithGLBuffers(GLBufferNames, interOpCLGL::ACQUIRE); }
  bool releaseGLBuffers(const std::vector<std::string>& GLBufferNames) { return interactWithGLBuffers(GLBufferNames, interOpCLGL::RELEASE); }

  bool mapAndSendBufferToDevice(std::string bufferName, const void* bufferPtr, size_t bufferSize);

  private:
  Context(bool profilingEnabled = true);
  ~Context() = default;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(const Context&&) = delete;

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