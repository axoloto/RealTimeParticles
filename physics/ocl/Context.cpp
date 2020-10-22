#pragma once

//#include "CL/cl2.hpp" // WIP
#include "windows.h" // WIP

//#include "CL/cl.h"
#include "Context.hpp"
#include <array>
#include <spdlog/spdlog.h>
#include <vector>

#define PROGRAM_FILE "C:\\Dev_perso\\boids\\physics\\ocl\\kernels\\boids.cl"

Core::CL::Context::Context()
    : m_preferredPlatformName("NVIDIA")
    , m_preferredDeviceType(CL_DEVICE_TYPE_GPU) {};

Core::CL::Context::~Context()
{
  clReleaseCommandQueue(cl_queue);
  clReleaseProgram(cl_program);
  clReleaseContext(cl_context);
}

bool Core::CL::Context::init()
{
  cl_int err;

  cl_uint numPlatforms;
  err = clGetPlatformIDs(1, nullptr, &numPlatforms);
  if (err != CL_SUCCESS)
  {
    printf("error when looking for platforms");
    return false;
  }

  std::vector<std::string> platformNames;
  cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
  clGetPlatformIDs(numPlatforms, platforms, nullptr);
  if (numPlatforms == 0)
  {
    printf("Couldn't retrieve any platform.");
    return false;
  }

  platformNames.resize(numPlatforms);
  for (int i = 0; i < numPlatforms; i++)
  {
    char data[1024];
    size_t retsize;
    err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(data), data, &retsize);
    if (err != CL_SUCCESS || retsize == 0)
    {
      printf("Couldn't find platform name.");
      return false;
    }

    platformNames[i] = std::string(data);
  }

  cl_platform = nullptr;
  for (int i = 0; i < numPlatforms; ++i)
  {
    if (platformNames[i].find(m_preferredPlatformName) != std::string::npos)
    {
      cl_platform = platforms[i];
      err = clGetDeviceIDs(cl_platform, m_preferredDeviceType, 1, &cl_device, nullptr);
      if (err != CL_SUCCESS || cl_device < 0)
      {
        printf("Couldn't find desired device");
        return false;
      }
      break;
    }
  }

  if (!cl_platform)
  {
    cl_platform = platforms[0];
    err = clGetDeviceIDs(cl_platform, CL_DEVICE_TYPE_ALL, 1, &cl_device, nullptr);
    if (err != CL_SUCCESS || cl_device < 0)
    {
      printf("Couldn't find device");
      return false;
    }
  }

  if (!isExtensionSupported(cl_device, "cl_khr_gl_sharing"))
  {
    printf("error, extension missing to do inter operation between opencl and opengl");
    return false;
  }

  cl_context_properties props[] = {
    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
    CL_CONTEXT_PLATFORM, (cl_context_properties)cl_platform,
    0
  };

  cl_context = clCreateContext(props, 1, &cl_device, NULL, NULL, &err);
  if (err != CL_SUCCESS)
  {
    cl_platform = platforms[0];
    err = clGetDeviceIDs(cl_platform, CL_DEVICE_TYPE_ALL, 1, &cl_device, nullptr);
    if (err != CL_SUCCESS || cl_device < 0)
    {
      printf("Couldn't find device");
      return false;
    }
    cl_context = clCreateContext(props, 1, &cl_device, NULL, NULL, &err);

    printf("error when creating context");
    return false;
  }

  FILE* program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;

  program_handle = fopen(PROGRAM_FILE, "rb");
  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);

  program_buffer = (char*)malloc(program_size + 1);
  program_buffer[program_size] = '\0';

  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  cl_program = clCreateProgramWithSource(cl_context, 1, (const char**)&program_buffer, &program_size, &err);
  if (err != CL_SUCCESS)
  {
    printf("error when creating program");
    return false;
  }
  free(program_buffer);

  const char options[] = "-DBOIDS_EFFECT_RADIUS_SQUARED=1000 -DBOIDS_MAX_STEERING=0.5f -DBOIDS_MAX_VELOCITY=5.0f -DABS_WALL_POS=250.0f -cl-denorms-are-zero -cl-fast-relaxed-math";
  err = clBuildProgram(cl_program, 1, &cl_device, options, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    clGetProgramBuildInfo(cl_program, cl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    program_log = (char*)malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(cl_program, cl_device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
    printf("%s\n", program_log);
    spdlog::error(program_log);
    free(program_log);
    return false;
  }

  cl_command_queue_properties properties = 0;
  if (m_kernelProfilingEnabled)
  {
    properties |= CL_QUEUE_PROFILING_ENABLE;
  }

  cl_queue = clCreateCommandQueue(cl_context, cl_device, properties, &err);
  if (err != CL_SUCCESS)
  {
    printf("error when creating queue");
    return false;
  }

  return true;
}

bool Core::CL::Context::isExtensionSupported(cl_device_id device, const char* extension)
{
  if (extension == NULL || extension[0] == '\0')
    return false;

  char* where = (char*)strchr(extension, ' ');
  if (where != NULL)
    return false;

  size_t extensionSize;
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);

  char* extensions = new char[extensionSize];
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, extensionSize, extensions, NULL);

  bool foundExtension = false;
  for (char* start = extensions;;)
  {
    where = (char*)strstr((const char*)start, extension);
    char* terminator = where + strlen(extension);

    if (*terminator == ' ' || *terminator == '\0' || *terminator == '\r' || *terminator == '\n')
    {
      foundExtension = true;
      break;
    }

    start = terminator;
  }

  delete[] extensions;

  return foundExtension;
}

// bool createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO);
// bool acquireGLBuffers(const std::vector<cl_mem>& GLBuffers);
// bool releaseGLBuffers(const std::vector<cl_mem>& GLBuffers);
// bool createKernels();
// void runKernel(cl_kernel kernel, double* profilingTimeMs = nullptr);
// void updateBoidsParamsInKernel();
