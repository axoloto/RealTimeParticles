#pragma once

#ifdef WIN32
#include "windows.h"
#else
#ifdef UNIX
#include "glx.h"
#endif
#endif

#include "Context.hpp"
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <vector>

Core::CL::Context::Context(std::string sourcePath, std::string specificBuildOptions, bool profilingEnabled)
    : m_sourceFilePath(sourcePath)
    , m_specificBuildOptions(specificBuildOptions)
    , m_isKernelProfilingEnabled(profilingEnabled)
    , m_init(false) {};

bool Core::CL::Context::init()
{
  if (m_init)
    return true;

  if (!findPlatforms())
    return false;

  if (!findGPUDevices())
    return false;

  if (!createContext())
    return false;

  if (!createAndBuildProgram())
    return false;

  if (!createCommandQueue())
    return false;

  m_init = true;
  return m_init;
}

bool Core::CL::Context::findPlatforms()
{
  spdlog::info("Searching for OpenCL platforms");

  cl::Platform::get(&m_allPlatforms);

  if (m_allPlatforms.empty())
  {
    spdlog::error("No OpenCL platform found");
    return false;
  }

  for (const auto& platform : m_allPlatforms)
  {
    std::string platformName;
    platform.getInfo(CL_PLATFORM_NAME, &platformName);
    spdlog::info("Found OpenCL platform : {}", platformName);
  }

  return true;
}

bool Core::CL::Context::findGPUDevices()
{
  spdlog::info("Searching for GPUs able to Interop OpenGL-OpenCL");

  for (const auto& platform : m_allPlatforms)
  {
    std::vector<cl::Device> GPUsOnPlatform;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &GPUsOnPlatform);

    if (GPUsOnPlatform.empty())
      continue;

    std::string platformName;
    platform.getInfo(CL_PLATFORM_NAME, &platformName);

    std::vector<cl::Device> GPUsOnPlatformWithInteropCLGL;

    for (const auto& GPU : GPUsOnPlatform)
    {
      std::string deviceName;
      GPU.getInfo(CL_DEVICE_NAME, &deviceName);

      std::string extensions;
      GPU.getInfo(CL_DEVICE_EXTENSIONS, &extensions);

      if (extensions.find("cl_khr_gl_sharing") != std::string::npos)
      {
        spdlog::info("Found GPU {} on platform {}", deviceName, platformName);
        GPUsOnPlatformWithInteropCLGL.push_back(GPU);
      }
    }

    if (!GPUsOnPlatformWithInteropCLGL.empty())
      m_allGPUsWithInteropCLGL.push_back(std::make_pair(platform, GPUsOnPlatformWithInteropCLGL));
  }

  if (m_allGPUsWithInteropCLGL.empty())
  {
    spdlog::error("No GPU found with Interop OpenCL-OpenGL extension");
    return false;
  }

  return true;
}

bool Core::CL::Context::createContext()
{
  // Looping to find the platform and the device used to display the application
  // We need to create our OpenCL context on those ones in order to
  // have InterOp OpenGL-OpenCL and share GPU memory buffers without transfers

  spdlog::info("Trying to create an OpenCL context");

  for (const auto& platformGPU : m_allGPUsWithInteropCLGL)
  {
    const auto platform = platformGPU.first;
    const auto GPUs = platformGPU.second;

#ifdef WIN32
    cl_context_properties props[] = {
      CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
      CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
      0
    };
#endif
#ifdef UNIX
    cl_context_properties props[] = {
      CL_GL_CONTEXT_KHR,
      (cl_context_properties)glXGetCurrentContext(),
      CL_GLX_DISPLAY_KHR,
      (cl_context_properties)glXGetCurrentDisplay(),
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)platform(), 0
    };
#endif

    for (const auto& GPU : GPUs)
    {
      cl_int err;
      cl_context = cl::Context(GPU, props, nullptr, nullptr, &err);
      if (err == CL_SUCCESS)
      {
        std::string platformName;
        platform.getInfo(CL_PLATFORM_NAME, &platformName);
        cl_platform = platform;

        std::string deviceName;
        GPU.getInfo(CL_DEVICE_NAME, &deviceName);
        cl_device = GPU;

        spdlog::info("Success! Created an OpenCL context with platform {} and GPU {}", platformName, deviceName);
        return true;
      }
    }
  }

  spdlog::error("Error while creating OpenCL context");
  return false;
}

bool Core::CL::Context::createAndBuildProgram()
{
  if (cl_context() == 0)
    return false;

  std::ifstream sourceFile(m_sourceFilePath);
  std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
  cl::Program::Sources source({ sourceCode });

  cl_program = cl::Program(cl_context, source);

  std::string options = m_specificBuildOptions + std::string(" -cl-denorms-are-zero -cl-fast-relaxed-math");
  cl_int err = cl_program.build({ cl_device }, options.c_str());
  if (err != CL_SUCCESS)
  {
    spdlog::error("Error while building OpenCL program : {}", cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_device));
    return false;
  }

  return true;
}

bool Core::CL::Context::createCommandQueue()
{
  if (cl_context() == 0 || cl_device() == 0)
    return false;

  cl_command_queue_properties properties = 0;
  if (m_isKernelProfilingEnabled)
  {
    properties |= CL_QUEUE_PROFILING_ENABLE;
  }

  cl_int err;
  cl_queue = cl::CommandQueue(cl_context, cl_device, properties, &err);
  if (err != CL_SUCCESS)
  {
    spdlog::error("Error when creating OpenCL queue");
    return false;
  }

  return true;
}

bool Core::CL::Context::createBuffer(std::string bufferName, size_t bufferSize, cl_mem_flags memoryFlags)
{
  if (!m_init)
    return false;

  cl_int err;

  if (m_buffersMap.find(bufferName) != m_buffersMap.end())
  {
    printf("error buffer already existing");
    return false;
  }

  auto buffer = cl::Buffer(cl_context, memoryFlags, bufferSize, nullptr, &err); //WIP device-only buffer for now

  if (err != CL_SUCCESS)
  {
    printf("error when creating buffer");
    return false;
  }

  m_buffersMap.insert(std::make_pair(bufferName, buffer));

  return true;
}

bool Core::CL::Context::createGLBuffer(std::string GLBufferName, unsigned int VBOIndex, cl_mem_flags memoryFlags)
{
  if (!m_init)
    return false;

  cl_int err;

  if (m_GLBuffersMap.find(GLBufferName) != m_GLBuffersMap.end())
  {
    printf("error GL buffer already existing");
    return false;
  }

  auto GLBuffer = cl::BufferGL(cl_context, memoryFlags, (cl_GLuint)VBOIndex, &err);

  if (err != CL_SUCCESS)
  {
    printf("error when creating GL buffer");
    return false;
  }

  m_GLBuffersMap.insert(std::make_pair(GLBufferName, GLBuffer));

  return true;
}

bool Core::CL::Context::createKernel(std::string kernelName, std::vector<std::string> bufferNames)
{
  // WIP Only taking buffer as args for now

  if (!m_init)
    return false;

  cl_int err;

  if (m_kernelsMap.find(kernelName) != m_kernelsMap.end())
  {
    printf("error kernel already existing");
    return false;
  }

  auto kernel = cl::Kernel(cl_program, kernelName.c_str(), &err);

  if (err != CL_SUCCESS)
  {
    printf("error when creating kernel");
    return false;
  }

  for (cl_uint i = 0; i < bufferNames.size(); ++i)
  {
    auto it = m_buffersMap.find(bufferNames[i]);
    auto itGL = m_GLBuffersMap.find(bufferNames[i]);
    if (it != m_buffersMap.end())
    {
      kernel.setArg(i, it->second);
    }
    else if (itGL != m_GLBuffersMap.end())
    {
      kernel.setArg(i, itGL->second);
    }
    else
    {
      printf("error kernel buffer arg not existing");
      return false;
    }
  }

  m_kernelsMap.insert(std::make_pair(kernelName, kernel));

  return true;
}

bool Core::CL::Context::setKernelArg(std::string kernelName, cl_uint argIndex, size_t argSize, const void* value)
{
  if (!m_init)
    return false;

  auto it = m_kernelsMap.find(kernelName);
  if (it == m_kernelsMap.end())
  {
    printf("error kernel not existing");
    return false;
  }

  auto kernel = it->second;
  kernel.setArg(argIndex, argSize, value);

  return true;
}

bool Core::CL::Context::runKernel(std::string kernelName, size_t numWorkItems) //WIP 1D Global
{
  if (!m_init)
    return false;

  auto it = m_kernelsMap.find(kernelName);
  if (it == m_kernelsMap.end())
  {
    printf("error kernel not existing");
    return false;
  }

  cl::Event event;
  size_t global1D = (numWorkItems > 10) ? numWorkItems - (numWorkItems % 10) : numWorkItems; //WIP
  cl::NDRange global(global1D);
  cl_queue.enqueueNDRangeKernel(it->second, cl::NullRange, global, cl::NullRange, nullptr, &event);

  if (m_isKernelProfilingEnabled)
  {
    cl_int err;

    err = cl_queue.flush();
    if (err != CL_SUCCESS)
      printf("error when flushing opencl run");

    err = cl_queue.finish();
    if (err != CL_SUCCESS)
      printf("error when finishing opencl run");

    cl_ulong start = 0, end = 0;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    //the resolution of the events is 1e-09 sec
    double profilingTimeMs = (double)((cl_double)(end - start) * (1e-06));

    //printf("%s %f ms \n", kernelName.c_str(), profilingTimeMs);
  }

  return true;
}

bool Core::CL::Context::interactWithGLBuffers(const std::vector<std::string>& GLBufferNames, interOpCLGL interaction)
{
  if (!m_init)
    return false;

  std::vector<cl::Memory> GLBuffers;

  for (const auto& GLBufferName : GLBufferNames)
  {
    auto it = m_GLBuffersMap.find(GLBufferName);
    if (it == m_GLBuffersMap.end())
    {
      printf("error GL buffer not existing");
      return false;
    }
    else
    {
      GLBuffers.push_back(it->second);
    }
  }

  cl_int err = (interaction == interOpCLGL::ACQUIRE) ? cl_queue.enqueueAcquireGLObjects(&GLBuffers) : cl_queue.enqueueReleaseGLObjects(&GLBuffers);
  if (err != CL_SUCCESS)
  {
    printf("error when interacting with GL buffers");
    return false;
  }

  return true;
}

bool Core::CL::Context::mapAndSendBufferToDevice(std::string bufferName, const void* bufferPtr, size_t bufferSize)
{
  if (!m_init || bufferPtr == nullptr)
    return false;

  auto it = m_buffersMap.find(bufferName);
  if (it == m_buffersMap.end())
  {
    printf("error buffer not existing");
    return false;
  }

  cl_int err;
  void* mappedMemory = cl_queue.enqueueMapBuffer(it->second, CL_TRUE, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &err);
  if (err < 0)
  {
    printf("Couldn't map the buffer to host memory");
    return false;
  }
  memcpy(mappedMemory, bufferPtr, bufferSize);
  err = cl_queue.enqueueUnmapMemObject(it->second, mappedMemory);
  if (err < 0)
  {
    printf("Couldn't unmap the buffer");
    return false;
  }

  return true;
}