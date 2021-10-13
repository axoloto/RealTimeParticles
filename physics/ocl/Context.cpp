#pragma once

#ifdef WIN32
#include "windows.h"
#else
#ifdef UNIX
#include "glx.h"
#endif
#endif

#include "Context.hpp"
#include "ErrorCode.hpp"
#include "Logging.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>


Physics::CL::Context& Physics::CL::Context::Get()
{
  static Context context;
  return context;
}

Physics::CL::Context::Context()
    : m_isKernelProfilingEnabled(false)
    , m_init(false)
{
  if (!findPlatforms())
    return;

  if (!findGPUDevices())
    return;

  if (!createContext())
    return;

  if (!createCommandQueue())
    return;

  m_init = true;
}

bool Physics::CL::Context::findPlatforms()
{
  LOG_INFO("Searching for OpenCL platforms");

  cl::Platform::get(&m_allPlatforms);

  if (m_allPlatforms.empty())
  {
    LOG_ERROR("No OpenCL platform found");
    return false;
  }

  for (const auto& platform : m_allPlatforms)
  {
    std::string platformName;
    platform.getInfo(CL_PLATFORM_NAME, &platformName);
    LOG_INFO("Found OpenCL platform : {}", platformName);
  }

  return true;
}

bool Physics::CL::Context::findGPUDevices()
{
  LOG_INFO("Searching for GPUs able to Interop OpenGL-OpenCL");

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
        LOG_INFO("Found GPU {} on platform {}", deviceName, platformName);
        GPUsOnPlatformWithInteropCLGL.push_back(GPU);
      }
    }

    if (!GPUsOnPlatformWithInteropCLGL.empty())
      m_allGPUsWithInteropCLGL.push_back(std::make_pair(platform, GPUsOnPlatformWithInteropCLGL));
  }

  if (m_allGPUsWithInteropCLGL.empty())
  {
    LOG_ERROR("No GPU found with Interop OpenCL-OpenGL extension");
    return false;
  }

  return true;
}

bool Physics::CL::Context::createContext()
{
  // Looping to find the platform and the device used to display the application
  // We need to create our OpenCL context on those ones in order to
  // have InterOp OpenGL-OpenCL and share GPU memory buffers without transfers

  LOG_INFO("Trying to create an OpenCL context");

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

        LOG_INFO("Success! Created an OpenCL context with platform {} and GPU {}", platformName, deviceName);
        return true;
      }
    }
  }

  LOG_ERROR("Error while creating OpenCL context");
  return false;
}

bool Physics::CL::Context::createCommandQueue()
{
  if (cl_context() == 0 || cl_device() == 0)
    return false;

  cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

  cl_int err;
  cl_queue = cl::CommandQueue(cl_context, cl_device, properties, &err);
  if (err != CL_SUCCESS)
  {
    LOG_ERROR("Cannot create OpenCL queue");
    return false;
  }
  return true;
}

bool Physics::CL::Context::release()
{
  if (!m_init)
    return true;

  finishTasks();

  LOG_DEBUG("Physics::CL::Context::release - Context has been cleaned");

  m_programsMap.clear();
  m_kernelsMap.clear();
  m_buffersMap.clear();
  m_GLBuffersMap.clear();
  m_imagesMap.clear();

  return true;
}

bool Physics::CL::Context::finishTasks()
{
  cl_int err = cl_queue.flush();
  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Cannot flush queue");
    return false;
  }

  err = cl_queue.finish();
  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Cannot finish queue");
    return false;
  }

  LOG_DEBUG("Explicitly flushed and finished OpenCL device queue");
  return true;
}

bool Physics::CL::Context::createProgram(std::string programName, std::vector<std::string> sourceNames, std::string specificBuildOptions)
{
  if (!m_init)
    return false;

  cl::Program::Sources sources;
  for (const auto& sourceName : sourceNames)
  {
    // Little hack to make it work from both installer and local build
    std::ifstream sourceFile(".\\kernels\\" + sourceName);

    if (!sourceFile.is_open())
      sourceFile = std::ifstream(Utils::GetSrcDir() + "\\physics\\ocl\\kernels\\" + sourceName);

    if (!sourceFile.is_open())
      LOG_ERROR("Cannot find kernel file {}", sourceName);

    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    sources.push_back(sourceCode);
  }

  auto program = cl::Program(cl_context, sources);

  std::string options = specificBuildOptions + std::string(" -cl-denorms-are-zero -cl-fast-relaxed-math");
  cl_int err = program.build({ cl_device }, options.c_str());
  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Cannot build program");
    LOG_ERROR(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl_device));
    throw std::runtime_error(" Exiting Program ");
    return false;
  }

  m_programsMap.insert(std::make_pair(programName, program));

  return true;
}

bool Physics::CL::Context::createBuffer(std::string bufferName, size_t bufferSize, cl_mem_flags memoryFlags)
{
  if (!m_init)
    return false;

  cl_int err;

  if (m_buffersMap.find(bufferName) != m_buffersMap.end())
  {
    LOG_ERROR("Buffer {} already existing", bufferName);
    return false;
  }

  auto buffer = cl::Buffer(cl_context, memoryFlags, bufferSize, nullptr, &err);

  if (err != CL_SUCCESS)
  {
    LOG_ERROR("Cannot create buffer {}", bufferName);
    return false;
  }

  m_buffersMap.insert(std::make_pair(bufferName, buffer));

  return true;
}

bool Physics::CL::Context::createImage2D(std::string name, imageSpecs specs, cl_mem_flags memoryFlags)
{
  if (!m_init)
    return false;

  cl_int err;

  if (m_imagesMap.find(name) != m_imagesMap.end())
  {
    LOG_ERROR("Image {} already existing", name);
    return false;
  }

  cl::ImageFormat format(specs.channelOrder, specs.channelType);

  auto image = cl::Image2D(cl_context, memoryFlags, format, specs.width, specs.height, 0, nullptr, &err);

  if (err != CL_SUCCESS)
  {
    LOG_ERROR("Cannot create image {}", name);
    return false;
  }

  m_imagesMap.insert(std::make_pair(name, image));

  return true;
}

bool Physics::CL::Context::loadBufferFromHost(std::string bufferName, size_t offset, size_t sizeToFill, const void* hostPtr)
{
  if (!m_init)
    return false;

  cl_int err;

  cl::Buffer destBuffer;

  auto itSrc = m_buffersMap.find(bufferName);
  if (itSrc == m_buffersMap.end())
  {
    auto itSrcGL = m_GLBuffersMap.find(bufferName);

    if (itSrcGL == m_GLBuffersMap.end())
    {
      LOG_ERROR("Buffer {} not existing", bufferName);
      return false;
    }
    else
    {
      destBuffer = itSrcGL->second;
    }
  }
  else
    destBuffer = itSrc->second;

  err = cl_queue.enqueueWriteBuffer(destBuffer, CL_TRUE, offset, sizeToFill, hostPtr);

  if (err != CL_SUCCESS)
  {
    LOG_ERROR("Cannot load buffer {}", bufferName);
    return false;
  }

  return true;
}

bool Physics::CL::Context::unloadBufferFromDevice(std::string bufferName, size_t offset, size_t sizeToFill, void* hostPtr)
{
  if (!m_init)
    return false;

  cl_int err;

  cl::Buffer srcBuffer;

  auto itSrc = m_buffersMap.find(bufferName);
  if (itSrc == m_buffersMap.end())
  {
    auto itSrcGL = m_GLBuffersMap.find(bufferName);

    if (itSrcGL == m_GLBuffersMap.end())
    {
      LOG_ERROR("Buffer {} not existing", bufferName);
      return false;
    }
    else
    {
      srcBuffer = itSrcGL->second;
    }
  }
  else
    srcBuffer = itSrc->second;

  err = cl_queue.enqueueReadBuffer(srcBuffer, CL_TRUE, offset, sizeToFill, hostPtr);

  if (err != CL_SUCCESS)
  {
    LOG_ERROR("Cannot unload buffer {}", bufferName);
    return false;
  }

  return true;
}

bool Physics::CL::Context::swapBuffers(std::string bufferNameA, std::string bufferNameB)
{
  if (!m_init)
    return false;

  auto& itA = m_buffersMap.find(bufferNameA);
  if (itA == m_buffersMap.end())
  {
    LOG_ERROR("Cannot swap buffers, buffer {} not existing", bufferNameA);
    return false;
  }

  auto& itB = m_buffersMap.find(bufferNameB);
  if (itB == m_buffersMap.end())
  {
    LOG_ERROR("Cannot swap buffers, buffer {} not existing", bufferNameB);
    return false;
  }

  auto tempBuffer = itA->second;
  itA->second = itB->second;
  itB->second = tempBuffer;

  return true;
}

bool Physics::CL::Context::copyBuffer(std::string srcBufferName, std::string dstBufferName)
{
  if (!m_init)
    return false;

  cl_int err;

  cl::Buffer srcBuffer;

  auto& itSrc = m_buffersMap.find(srcBufferName);

  if (itSrc == m_buffersMap.end())
  {
    auto itSrcGL = m_GLBuffersMap.find(srcBufferName);

    if (itSrcGL == m_GLBuffersMap.end())
    {
      LOG_ERROR("Cannot copy buffers, source buffer {} not existing", srcBufferName);
      return false;
    }
    else
    {
      srcBuffer = itSrcGL->second;
    }
  }
  else
  {
    srcBuffer = itSrc->second;
  }

  auto& itDst = m_buffersMap.find(dstBufferName);
  if (itDst == m_buffersMap.end())
  {
    LOG_ERROR("Cannot copy buffers, destination buffer {} not existing", dstBufferName);
    return false;
  }

  cl::Buffer dstBuffer = itDst->second;
  size_t dstBufferSize;
  err = dstBuffer.getInfo(CL_MEM_SIZE, &dstBufferSize);

  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Cannot get size from buffer " + dstBufferName);
    return false;
  }

  size_t srcBufferSize;
  err = srcBuffer.getInfo(CL_MEM_SIZE, &srcBufferSize);

  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Cannot get size from buffer" + srcBufferName);
    return false;
  }

  if (dstBufferSize > srcBufferSize)
  {
    LOG_ERROR("Source buffer {} with size {} is smaller than destination buffer {} with size {} ", srcBufferName, srcBufferSize, dstBufferName, dstBufferSize);
    return false;
  }

  // Only copying the amount of data which can fit into the destination buffer
  err = cl_queue.enqueueCopyBuffer(srcBuffer, dstBuffer, 0, 0, dstBufferSize);

  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Cannot copy buffer " + srcBufferName + " to buffer " + dstBufferName);
    return false;
  }

  return true;
}

bool Physics::CL::Context::createGLBuffer(std::string GLBufferName, unsigned int VBOIndex, cl_mem_flags memoryFlags)
{
  if (!m_init)
    return false;

  cl_int err;

  if (m_GLBuffersMap.find(GLBufferName) != m_GLBuffersMap.end())
  {
    LOG_ERROR("GL buffer {} already existing", GLBufferName);
    return false;
  }

  auto GLBuffer = cl::BufferGL(cl_context, memoryFlags, (cl_GLuint)VBOIndex, &err);

  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Cannot create GL buffer " + GLBufferName);
    return false;
  }

  m_GLBuffersMap.insert(std::make_pair(GLBufferName, GLBuffer));

  return true;
}

bool Physics::CL::Context::createKernel(std::string programName, std::string kernelName, std::vector<std::string> argNames)
{
  // WIP Only taking buffer as args for now

  if (!m_init)
    return false;

  cl_int err;

  if (m_programsMap.find(programName) == m_programsMap.end())
  {
    LOG_ERROR("OpenCL program not existing {}", programName);
    return false;
  }

  if (m_kernelsMap.find(kernelName) != m_kernelsMap.end())
  {
    LOG_ERROR("OpenCL kernel already existing {}", kernelName);
    return false;
  }

  auto kernel = cl::Kernel(m_programsMap.at(programName), kernelName.c_str(), &err);

  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Cannot create kernel " + kernelName);
    return false;
  }

  for (cl_uint i = 0; i < argNames.size(); ++i)
  {
    if (argNames[i].empty())
      continue;

    auto it = m_buffersMap.find(argNames[i]);
    auto itGL = m_GLBuffersMap.find(argNames[i]);
    auto itIm = m_imagesMap.find(argNames[i]);
    if (it != m_buffersMap.end())
    {
      kernel.setArg(i, it->second);
    }
    else if (itGL != m_GLBuffersMap.end())
    {
      kernel.setArg(i, itGL->second);
    }
    else if (itIm != m_imagesMap.end())
    {
      kernel.setArg(i, itIm->second);
    }
    else
    {
      LOG_ERROR("For kernel {} arg not existing {}", kernelName, argNames[i]);
      return false;
    }
  }

  m_kernelsMap.insert(std::make_pair(kernelName, kernel));

  return true;
}

bool Physics::CL::Context::setKernelArg(std::string kernelName, cl_uint argIndex, size_t argSize, const void* value)
{
  if (!m_init)
    return false;

  auto it = m_kernelsMap.find(kernelName);
  if (it == m_kernelsMap.end())
  {
    LOG_ERROR("Cannot set arg {} for unexisting Kernel {}", argIndex, kernelName);
    return false;
  }

  auto kernel = it->second;
  cl_int err = kernel.setArg(argIndex, argSize, value);

  if (err != CL_SUCCESS)
  {
    LOG_ERROR("Cannot set arg {} for kernel {} ", argIndex, kernelName);
    return false;
  }

  return true;
}

bool Physics::CL::Context::setKernelArg(std::string kernelName, cl_uint argIndex, const std::string& argName)
{
  if (!m_init)
    return false;

  auto itK = m_kernelsMap.find(kernelName);
  if (itK == m_kernelsMap.end())
  {
    LOG_ERROR("Cannot set arg {} for unexisting Kernel {}", argName, kernelName);
    return false;
  }

  auto kernel = itK->second;

  auto itB = m_buffersMap.find(argName);
  auto itBGL = m_GLBuffersMap.find(argName);
  auto itIm = m_imagesMap.find(argName);

  if (itB != m_buffersMap.end())
  {
    kernel.setArg(argIndex, itB->second);
  }
  else if (itBGL != m_GLBuffersMap.end())
  {
    kernel.setArg(argIndex, itBGL->second);
  }
  else if (itIm != m_imagesMap.end())
  {
    kernel.setArg(argIndex, itIm->second);
  }
  else
  {
    LOG_ERROR("For kernel {} arg not existing {}", kernelName, argName);
    return false;
  }

  return true;
}

bool Physics::CL::Context::runKernel(std::string kernelName, size_t numGlobalWorkItems, size_t numLocalWorkItems)
{
  if (!m_init)
    return false;

  auto it = m_kernelsMap.find(kernelName);
  if (it == m_kernelsMap.end())
  {
    LOG_ERROR("Cannot run unexisting Kernel {}", kernelName);
    return false;
  }

  cl::Event event;
  cl::NDRange global(numGlobalWorkItems);
  cl::NDRange local = (numLocalWorkItems > 0) ? cl::NDRange(numLocalWorkItems) : cl::NullRange;

  cl_int err;

  err = cl_queue.enqueueNDRangeKernel(it->second, cl::NullRange, global, local, nullptr, &event);
  if (err != CL_SUCCESS)
  {
    CL_ERROR(err, "Failure of kernel " + kernelName + " while running");
    return false;
  }

  if (m_isKernelProfilingEnabled)
  {
    if (!finishTasks())
      return false;

    cl_ulong start = 0, end = 0;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    //the resolution of the events is 1e-09 sec
    double profilingTimeMs = (double)((cl_double)(end - start) * (1e-06));

    //if (profilingTimeMs > 1.0)
    LOG_INFO("Profiling kernel {} : {} ms", kernelName, profilingTimeMs);
  }

  return true;
}

bool Physics::CL::Context::interactWithGLBuffers(const std::vector<std::string>& GLBufferNames, interOpCLGL interaction)
{
  if (!m_init)
    return false;

  std::vector<cl::Memory> GLBuffers;

  for (const auto& GLBufferName : GLBufferNames)
  {
    auto it = m_GLBuffersMap.find(GLBufferName);
    if (it == m_GLBuffersMap.end())
    {
      LOG_ERROR("error GL buffer not existing");
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
    CL_ERROR(err, "Cannot interact with GL buffers");
    return false;
  }
  else
  {
    std::string allNames;
    std::for_each(GLBufferNames.cbegin(), GLBufferNames.cend(), [&](const std::string& name)
        { return allNames += name + " "; });
    LOG_DEBUG(interaction == interOpCLGL::ACQUIRE ? "GL buffers acquired {}" : "GL buffers released {}", allNames);
  }

  // Must flush and finish queue to make sure GL buffers have been released
  if (interaction == interOpCLGL::RELEASE)
    finishTasks();

  return true;
}

bool Physics::CL::Context::mapAndSendBufferToDevice(std::string bufferName, const void* bufferPtr, size_t bufferSize)
{
  if (!m_init || bufferPtr == nullptr)
    return false;

  auto it = m_buffersMap.find(bufferName);
  if (it == m_buffersMap.end())
  {
    LOG_ERROR("error buffer not existing");
    return false;
  }

  cl_int err;
  void* mappedMemory = cl_queue.enqueueMapBuffer(it->second, CL_TRUE, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &err);
  if (err < 0)
  {
    CL_ERROR(err, "Cannot map buffer " + bufferName + " to host memory");
    return false;
  }
  memcpy(mappedMemory, bufferPtr, bufferSize);
  err = cl_queue.enqueueUnmapMemObject(it->second, mappedMemory);
  if (err < 0)
  {
    CL_ERROR(err, "Cannot unmap buffer" + bufferName);
    return false;
  }

  return true;
}

std::string Physics::CL::Context::getPlatformName() const
{
  std::string platformName;
  cl_platform.getInfo(CL_PLATFORM_NAME, &platformName);
  return platformName;
}

std::string Physics::CL::Context::getDeviceName() const
{
  std::string deviceName;
  cl_device.getInfo(CL_DEVICE_NAME, &deviceName);
  return deviceName;
}