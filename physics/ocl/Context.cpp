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
#include <spdlog/spdlog.h>
#include <vector>

Core::CL::Context::Context(std::string sourcePath, std::string specificBuildOptions)
    : m_preferredPlatformName("NVIDIA")
    , m_sourceFilePath(sourcePath)
    , m_specificBuildOptions(specificBuildOptions) {};

bool Core::CL::Context::init()
{
  bool nextStep = false;

  nextStep = findPlatform();
  if (!nextStep)
    return false;

  nextStep = findDevice();
  if (!nextStep)
    return false;

  nextStep = createContext();
  if (!nextStep)
    return false;

  nextStep = createAndBuildProgram();
  if (!nextStep)
    return false;

  nextStep = createCommandQueue();
  return nextStep;
}

bool Core::CL::Context::findPlatform()
{
  std::vector<cl::Platform> allPlatforms;
  cl::Platform::get(&allPlatforms);

  if (allPlatforms.empty())
  {
    printf("No OpenCL platform found.");
    return false;
  }

  for (const auto& platform : allPlatforms)
  {
    std::string namePlatform;
    platform.getInfo(CL_PLATFORM_NAME, &namePlatform);
    if (namePlatform.find(m_preferredPlatformName) != std::string::npos)
    {
      printf("Found desired platform.");
      cl_Platform = platform;
      break;
    }
  }

  if (cl_Platform() == 0)
  {
    printf("Desired platform not found.");
    return false;
  }

  return true;
}

bool Core::CL::Context::findDevice()
{
  if (cl_Platform() == 0)
    return false;

  // Looking only for GPUs to ensure OpenGL-CL interop
  std::vector<cl::Device> allGPUsOnPlatform;
  cl_Platform.getDevices(CL_DEVICE_TYPE_GPU, &allGPUsOnPlatform);
  if (allGPUsOnPlatform.empty())
  {
    printf("No GPUs found on selected platform.");
    return false;
  }

  for (const auto& GPU : allGPUsOnPlatform)
  {
    std::string extensions;
    GPU.getInfo(CL_DEVICE_EXTENSIONS, &extensions);

    if (extensions.find("cl_khr_gl_sharing") != std::string::npos)
    {
      cl_Device = GPU;
      printf("Found GPU with GL extension on selected platform.");
      return true;
    }
  }

  printf("No GPUs found with GL extension on selected platform.");
  return false;
}

bool Core::CL::Context::createContext()
{
#ifdef WIN32
  cl_context_properties props[] = {
    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
    CL_CONTEXT_PLATFORM, (cl_context_properties)cl_Platform(),
    0
  };
#else
#ifdef UNIX
  cl_context_properties props[] = {
    CL_GL_CONTEXT_KHR,
    (cl_context_properties)glXGetCurrentContext(),
    CL_GLX_DISPLAY_KHR,
    (cl_context_properties)glXGetCurrentDisplay(),
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)cl_Platform(), 0
  };
#endif
#endif
  cl_int err;
  cl_Context = cl::Context(cl_Device, nullptr, nullptr, nullptr, &err);
  // cl_Context = cl::Context(cl_Device, props, nullptr, nullptr, &err); //WIP
  if (err != CL_SUCCESS)
  {
    printf("error while creating context");
    return false;
  }

  return true;
}

bool Core::CL::Context::createAndBuildProgram()
{
  std::ifstream sourceFile(m_sourceFilePath);
  std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
  cl::Program::Sources source({ sourceCode });

  cl_Program = cl::Program(cl_Context, source);

  std::string options = m_specificBuildOptions + std::string("-cl-denorms-are-zero -cl-fast-relaxed-math");
  cl_int err = cl_Program.build({ cl_Device }, options.c_str());
  if (err != CL_SUCCESS)
  {
    printf("%s\n", cl_Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_Device));
    return false;
  }

  return true;
}

bool Core::CL::Context::createCommandQueue()
{
  cl_command_queue_properties properties = 0;
  if (m_kernelProfilingEnabled)
  {
    properties |= CL_QUEUE_PROFILING_ENABLE;
  }

  cl_int err;
  cl_Queue = cl::CommandQueue(cl_Context, cl_Device, properties, &err);
  if (err != CL_SUCCESS)
  {
    printf("error when creating queue");
    return false;
  }

  return true;
}

// bool createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
// bool acquireGLBuffers(const std::vector<cl_mem>& GLBuffers);
// bool releaseGLBuffers(const std::vector<cl_mem>& GLBuffers);
// bool createKernels();
// void runKernel(cl_kernel kernel, double* profilingTimeMs = nullptr);
// void updateBoidsParamsInKernel();