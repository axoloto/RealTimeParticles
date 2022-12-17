#pragma once

#include "opencl.hpp"

#include <map>
#include <string>
#include <vector>

namespace Physics
{
namespace CL
{
struct imageSpecs
{
  cl_channel_order channelOrder;
  cl_channel_type channelType;
  size_t width;
  size_t height;
};

class Context
{
  public:
  static Context& Get();

  // Check if the context has been instantiated
  bool isInit() const { return m_init; }
  // Release every programs and kernels/buffers/datas on GPU side
  bool release();

  // Send all the tasks to device queue and wait for them to be complete
  bool finishTasks();

  bool isProfiling() const { return m_isKernelProfilingEnabled; }
  void enableProfiler(bool enable) { m_isKernelProfilingEnabled = enable; }

  bool createProgram(std::string name, std::vector<std::string> sourceNames, std::string specificBuildOptions);
  bool createProgram(std::string name, std::string sourceName, std::string specificBuildOptions) { return createProgram(name, std::vector<std::string>({ sourceName }), specificBuildOptions); }
  bool createGLBuffer(std::string name, unsigned int VBOIndex, cl_mem_flags memoryFlags);
  bool createBuffer(std::string name, size_t bufferSize, cl_mem_flags memoryFlags);
  bool createImage2D(std::string name, imageSpecs specs, cl_mem_flags memoryFlags);
  bool loadBufferFromHost(std::string name, size_t offset, size_t sizeToFill, const void* hostPtr);
  bool unloadBufferFromDevice(std::string name, size_t offset, size_t sizeToFill, void* hostPtr);
  bool swapBuffers(std::string bufferNameA, std::string bufferNameB);
  bool copyBuffer(std::string srcBufferName, std::string dstBufferName);
  bool createKernel(std::string programName, std::string kernelName, std::vector<std::string> argNames);
  bool setKernelArg(std::string kernelName, cl_uint argIndex, size_t argSize, const void* value);
  bool setKernelArg(std::string kernelName, cl_uint argIndex, const std::string& bufferName);
  bool runKernel(std::string kernelName, size_t numFlobalWorkItems, size_t numLocalWorkItems = 0);

  bool acquireGLBuffers(const std::vector<std::string>& GLBufferNames) { return interactWithGLBuffers(GLBufferNames, interOpCLGL::ACQUIRE); }
  bool releaseGLBuffers(const std::vector<std::string>& GLBufferNames) { return interactWithGLBuffers(GLBufferNames, interOpCLGL::RELEASE); }

  bool mapAndSendBufferToDevice(std::string bufferName, const void* bufferPtr, size_t bufferSize);

  std::string getPlatformName() const;
  std::string getDeviceName() const;

  private:
  Context();
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
  std::map<std::string, cl::Image2D> m_imagesMap;

  bool m_isKernelProfilingEnabled;

  bool m_init;

  std::vector<cl::Platform> m_allPlatforms;
  std::vector<std::pair<cl::Platform, std::vector<cl::Device>>> m_allGPUsWithInteropCLGL;
};
} //CL
} //Core