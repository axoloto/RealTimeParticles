#include "CL/cl_gl.h" // WIP
#include "OCLBoids.hpp"
#include "windows.h" // WIP
#include <spdlog/spdlog.h>

using namespace Core;

#define PROGRAM_FILE "C:\\Dev_perso\\boids\\physics\\ocl\\kernels\\boids.cl"

#define KERNEL_RANDOM_POS_FUNC "randPosVerts"
#define KERNEL_BOIDS_RULES_FUNC "applyBoidsRules"
#define KERNEL_UPDATE_POS_FUNC "updatePosVerts"
#define KERNEL_COLOR_FUNC "colorVerts"

static bool isOCLExtensionSupported(cl_device_id device, const char* extension);

OCLBoids::OCLBoids(int boxSize, int numEntities, unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO)
    : Boids(boxSize, numEntities)
    , m_init(false)
{
  if (initOpenCL())
  {
    createBuffers(pointCloudCoordVBO, pointCloudColorVBO);
    createKernels();

    m_init = true;

    acquireGLBuffers({ cl_colorBuff, cl_posBuff });
    runKernel(cl_colorKernel);
    runKernel(cl_initPosKernel);
    releaseGLBuffers({ cl_colorBuff, cl_posBuff });
  }
}

OCLBoids::~OCLBoids()
{
  clReleaseKernel(cl_colorKernel);
  clReleaseMemObject(cl_colorBuff);

  clReleaseKernel(cl_initPosKernel);
  clReleaseMemObject(cl_posBuff);

  clReleaseKernel(cl_boidsRulesKernel);
  clReleaseMemObject(cl_accBuff);
  clReleaseMemObject(cl_velBuff);

  clReleaseCommandQueue(cl_queue);
  clReleaseProgram(cl_program);
  clReleaseContext(cl_context);
}

void OCLBoids::updatePhysics()
{
  acquireGLBuffers({ cl_posBuff });
  runKernel(cl_boidsRulesKernel);
  runKernel(cl_updatePosKernel);
  releaseGLBuffers({ cl_posBuff });
}

bool OCLBoids::initOpenCL()
{
  cl_int err;

  cl_uint numPlatforms;
  err = clGetPlatformIDs(1, nullptr, &numPlatforms);
  if (err != CL_SUCCESS)
  {
    printf("error when looking for platforms");
    return false;
  }

  cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
  clGetPlatformIDs(numPlatforms, platforms, nullptr);
  for (int i = 0; i < numPlatforms; i++)
  {
    char data[1024];
    size_t retsize;
    err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(data), data, &retsize);
    if (err != CL_SUCCESS)
    {
      printf("Couldn't find platform name.");
      return false;
    }

    if (retsize > 0)
    {
      std::string strData = data;
      if (strData.find("NVIDIA") != std::string::npos)
      {
        cl_platform = platforms[i];
        err = clGetDeviceIDs(cl_platform, CL_DEVICE_TYPE_GPU, 1, &cl_device, nullptr);
        if (err != CL_SUCCESS)
        {
          printf("Couldn't find NVIDIA GPU");
          return false;
        }
        printf("Found NVIDIA GPU");
        break;
      }
    }
  }

  if (cl_device < 0)
  {
    printf("Couldn't find NVIDIA GPU.");
    return false;
  }

  if (!isOCLExtensionSupported(cl_device, "cl_khr_gl_sharing"))
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

  err = clBuildProgram(cl_program, 1, &cl_device, NULL, NULL, NULL);
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

  cl_queue = clCreateCommandQueue(cl_context, cl_device, 0, &err);
  if (err != CL_SUCCESS)
  {
    printf("error when creating queue");
    return false;
  }

  return true;
}

bool OCLBoids::createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO)
{
  cl_int err;

  cl_colorBuff = clCreateFromGLBuffer(cl_context, CL_MEM_WRITE_ONLY, pointCloudColorVBO, &err);
  if (err != CL_SUCCESS)
    printf("error when creating color buffer");

  cl_posBuff = clCreateFromGLBuffer(cl_context, CL_MEM_READ_WRITE, pointCloudCoordVBO, &err);
  if (err != CL_SUCCESS)
    printf("error when creating boids position buffer");

  size_t boidsBufferSize = 4 * NUM_MAX_ENTITIES * sizeof(float);
  //clGetMemObjectInfo(cl_posBuff, CL_MEM_SIZE, sizeof(boidsBufferSize), &boidsBufferSize, NULL);
  //if (boidsBufferSize == 0)
  //  return false; //WIP

  cl_velBuff = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, boidsBufferSize, nullptr, &err);
  if (err != CL_SUCCESS)
    printf("error when creating boids velocity buffer");

  cl_accBuff = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, boidsBufferSize, nullptr, &err);
  if (err != CL_SUCCESS)
    printf("error when creating boids acceleration buffer");

  return true;
}

bool OCLBoids::createKernels()
{
  if (cl_colorBuff < 0 || cl_posBuff < 0 || cl_accBuff < 0 || cl_velBuff < 0)
    return false;

  cl_int err;

  cl_colorKernel = clCreateKernel(cl_program, KERNEL_COLOR_FUNC, &err);
  if (err != CL_SUCCESS)
    printf("error when creating color kernel");

  clSetKernelArg(cl_colorKernel, 0, sizeof(cl_mem), &cl_colorBuff);

  cl_initPosKernel = clCreateKernel(cl_program, KERNEL_RANDOM_POS_FUNC, &err);
  if (err != CL_SUCCESS)
    printf("error when creating random init position kernel");

  clSetKernelArg(cl_initPosKernel, 0, sizeof(cl_mem), &cl_posBuff);

  cl_boidsRulesKernel = clCreateKernel(cl_program, KERNEL_BOIDS_RULES_FUNC, &err);
  if (err != CL_SUCCESS)
    printf("error when creating boids rules kernel");

  clSetKernelArg(cl_boidsRulesKernel, 0, sizeof(cl_mem), &cl_posBuff);
  clSetKernelArg(cl_boidsRulesKernel, 1, sizeof(cl_mem), &cl_velBuff);
  clSetKernelArg(cl_boidsRulesKernel, 2, sizeof(cl_mem), &cl_accBuff);

  cl_updatePosKernel = clCreateKernel(cl_program, KERNEL_UPDATE_POS_FUNC, &err);
  if (err != CL_SUCCESS)
    printf("error when creating update position kernel");

  clSetKernelArg(cl_updatePosKernel, 0, sizeof(cl_mem), &cl_posBuff);
  clSetKernelArg(cl_updatePosKernel, 1, sizeof(cl_mem), &cl_velBuff);
  clSetKernelArg(cl_updatePosKernel, 2, sizeof(cl_mem), &cl_accBuff);

  return true;
}

bool OCLBoids::acquireGLBuffers(const std::vector<cl_mem>& GLBuffers)
{
  if (!m_init)
    return false;

  for (const auto& GLBuffer : GLBuffers)
  {
    cl_int err = clEnqueueAcquireGLObjects(cl_queue, 1, &GLBuffer, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
      printf("error when acquiring GL buffer");
  }

  return true;
}

bool OCLBoids::releaseGLBuffers(const std::vector<cl_mem>& GLBuffers)
{
  if (!m_init)
    return false;

  for (const auto& GLBuffer : GLBuffers)
  {
    cl_int err = clEnqueueReleaseGLObjects(cl_queue, 1, &GLBuffer, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
      printf("error when releasing GL buffer");
  }

  clFinish(cl_queue);

  return true;
}

void OCLBoids::runKernel(cl_kernel kernel)
{
  if (!m_init)
    return;

  size_t numWorkItems = NUM_MAX_ENTITIES;
  clEnqueueNDRangeKernel(cl_queue, kernel, 1, NULL, &numWorkItems, NULL, 0, NULL, NULL);
}

static bool isOCLExtensionSupported(cl_device_id device, const char* extension)
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