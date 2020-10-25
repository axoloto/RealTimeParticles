#include "Boids.hpp"
#include "windows.h" // WIP
#include <ctime>
#include <iostream>
#include <spdlog/spdlog.h>

using namespace Core;

//#define PROGRAM_FILE "C:\\Dev_perso\\boids\\physics\\ocl\\kernels\\boids.cl"

#define KERNEL_RANDOM_POS "randPosVerts"
#define KERNEL_BOIDS_RULES "applyBoidsRules"
#define KERNEL_UPDATE_POS_BOUNCING "updatePosVertsWithBouncingWalls"
#define KERNEL_UPDATE_POS_CYCLIC "updatePosVertsWithCyclicWalls"
#define KERNEL_UPDATE_VEL "updateVelVerts"
#define KERNEL_COLOR "colorVerts"

Boids::Boids(int numEntities, unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO)
    : Physics(numEntities)
    , m_scaleAlignment(2.0f)
    , m_scaleCohesion(0.7f)
    , m_scaleSeparation(1.2f)
    , m_activeTargets(false)
    , m_activeAlignment(true)
    , m_activeSeparation(true)
    , m_activeCohesion(true)
    , m_target({ 0.0f, 0.0f, 0.0f })
    , m_clContext(std::make_unique<CL::Context>(
          "physics\\ocl\\kernels\\boids.cl",
          "-DBOIDS_EFFECT_RADIUS_SQUARED=1000 -DBOIDS_MAX_STEERING=0.5f -DBOIDS_MAX_VELOCITY=5.0f -DABS_WALL_POS=250.0f"))
{
  if (m_clContext->init())
  {
    createBuffers(pointCloudCoordVBO, pointCloudColorVBO);
    createKernels();

    updateBoidsParamsInKernel();

    m_init = true;

    reset();
  }
}

Boids::~Boids()
{
  clReleaseKernel(cl_colorKernel);
  //clReleaseMemObject(cl_colorBuff);

  clReleaseKernel(cl_initPosKernel);
  //clReleaseMemObject(cl_posBuff);

  clReleaseKernel(cl_boidsRulesKernel);
  clReleaseMemObject(cl_accBuff);
  clReleaseMemObject(cl_velBuff);
}

void Boids::reset()
{
  double timeMs = 0.0;

  acquireGLBuffers({ cl_colorBuff, cl_posBuff });
  runKernel(cl_colorKernel, &timeMs);
  printf("cl_colorKernel %f ms \n", timeMs);

  cl_float dim = (m_dimension == Dimension::dim2D) ? 2.0f : 3.0f;
  clSetKernelArg(cl_initPosKernel, 2, sizeof(cl_float), &dim);
  runKernel(cl_initPosKernel, &timeMs);
  printf("cl_initPosKernel %f ms \n", timeMs);

  releaseGLBuffers({ cl_colorBuff, cl_posBuff });
}

void Boids::update()
{
  if (!m_init || m_pause)
    return;

  double timeMs = 0.0;
  acquireGLBuffers({ cl_posBuff });
  runKernel(cl_boidsRulesKernel, &timeMs);
  printf("cl_boidsRulesKernel %f ms \n", timeMs);
  runKernel(cl_updateVelKernel, &timeMs);
  printf("cl_updateVelKernel %f ms \n", timeMs);

  if (m_boundary == Boundary::CyclicWall)
  {
    runKernel(cl_updatePosCyclicWallsKernel, &timeMs);
    printf("cl_updatePosCyclicWallsKernel %f ms \n", timeMs);
  }
  else if (m_boundary == Boundary::BouncingWall)
  {
    runKernel(cl_updatePosBouncingWallsKernel, &timeMs);
    printf("cl_updatePosBouncingWallsKernel %f ms \n", timeMs);
  }
  else
  {
    printf("Something is wrong", timeMs);
  }

  releaseGLBuffers({ cl_posBuff });
}

void Boids::updateBoidsParamsInKernel()
{
  m_boidsParams.velocity = m_velocity;

  m_boidsParams.scaleCohesion = m_activeCohesion ? m_scaleCohesion : 0.0f;
  m_boidsParams.scaleAlignment = m_activeAlignment ? m_scaleAlignment : 0.0f;
  m_boidsParams.scaleSeparation = m_activeSeparation ? m_scaleSeparation : 0.0f;

  m_boidsParams.activeTarget = m_activeTargets ? 1 : 0;

  /////
  if (m_clContext->cl_queue < 0 || cl_boidsParamsBuff < 0)
    return;

  cl_int err;
  void* mappedMemory = clEnqueueMapBuffer(m_clContext->cl_queue, cl_boidsParamsBuff, CL_TRUE, CL_MAP_WRITE, 0, sizeof(m_boidsParams), 0, nullptr, nullptr, &err);
  if (err < 0)
  {
    printf("Couldn't map the buffer to host memory");
  }
  memcpy(mappedMemory, &m_boidsParams, sizeof(m_boidsParams));
  err = clEnqueueUnmapMemObject(m_clContext->cl_queue, cl_boidsParamsBuff, mappedMemory, 0, nullptr, nullptr);
  if (err < 0)
  {
    printf("Couldn't unmap the buffer");
  }
  ////
}

bool Boids::createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO)
{
  cl_int err;

  ////
  auto cl_context = m_clContext->cl_context;

  cl_colorBuff = clCreateFromGLBuffer(cl_context, CL_MEM_WRITE_ONLY, pointCloudColorVBO, &err);
  if (err != CL_SUCCESS)
    printf("error when creating color buffer");

  cl_posBuff = clCreateFromGLBuffer(cl_context, CL_MEM_READ_WRITE, pointCloudCoordVBO, &err);
  if (err != CL_SUCCESS)
    printf("error when creating boids position buffer");

  size_t boidsBufferSize = 4 * NUM_MAX_ENTITIES * sizeof(float);

  cl_velBuff = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, boidsBufferSize, nullptr, &err);
  if (err != CL_SUCCESS)
    printf("error when creating boids velocity buffer");

  cl_accBuff = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, boidsBufferSize, nullptr, &err);
  if (err != CL_SUCCESS)
    printf("error when creating boids acceleration buffer");

  cl_boidsParamsBuff = clCreateBuffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(m_boidsParams), nullptr, &err);
  /////

  return true;
}

bool Boids::createKernels()
{
  if (cl_colorBuff < 0 || cl_posBuff < 0 || cl_accBuff < 0 || cl_velBuff < 0 || cl_boidsParamsBuff < 0)
    return false;

  cl_int err;

  ////
  auto cl_program = m_clContext->cl_program;

  cl_colorKernel = clCreateKernel(cl_program, KERNEL_COLOR, &err);
  if (err != CL_SUCCESS)
    printf("error when creating color kernel");

  clSetKernelArg(cl_colorKernel, 0, sizeof(cl_mem), &cl_colorBuff);

  cl_initPosKernel = clCreateKernel(cl_program, KERNEL_RANDOM_POS, &err);
  if (err != CL_SUCCESS)
    printf("error when creating random init position kernel");

  clSetKernelArg(cl_initPosKernel, 0, sizeof(cl_mem), &cl_posBuff);
  clSetKernelArg(cl_initPosKernel, 1, sizeof(cl_mem), &cl_velBuff);

  cl_boidsRulesKernel = clCreateKernel(cl_program, KERNEL_BOIDS_RULES, &err);
  if (err != CL_SUCCESS)
    printf("error when creating boids rules kernel");

  clSetKernelArg(cl_boidsRulesKernel, 0, sizeof(cl_mem), &cl_posBuff);
  clSetKernelArg(cl_boidsRulesKernel, 1, sizeof(cl_mem), &cl_velBuff);
  clSetKernelArg(cl_boidsRulesKernel, 2, sizeof(cl_mem), &cl_accBuff);
  clSetKernelArg(cl_boidsRulesKernel, 3, sizeof(cl_mem), &cl_boidsParamsBuff);

  cl_updateVelKernel = clCreateKernel(cl_program, KERNEL_UPDATE_VEL, &err);
  if (err != CL_SUCCESS)
    printf("error when creating update position kernel");

  clSetKernelArg(cl_updateVelKernel, 0, sizeof(cl_mem), &cl_velBuff);
  clSetKernelArg(cl_updateVelKernel, 1, sizeof(cl_mem), &cl_accBuff);
  clSetKernelArg(cl_updateVelKernel, 2, sizeof(cl_mem), &cl_boidsParamsBuff);

  cl_updatePosBouncingWallsKernel = clCreateKernel(cl_program, KERNEL_UPDATE_POS_BOUNCING, &err);
  if (err != CL_SUCCESS)
    printf("error when creating update position kernel");

  clSetKernelArg(cl_updatePosBouncingWallsKernel, 0, sizeof(cl_mem), &cl_posBuff);
  clSetKernelArg(cl_updatePosBouncingWallsKernel, 1, sizeof(cl_mem), &cl_velBuff);

  cl_updatePosCyclicWallsKernel = clCreateKernel(cl_program, KERNEL_UPDATE_POS_CYCLIC, &err);
  if (err != CL_SUCCESS)
    printf("error when creating update position kernel");

  clSetKernelArg(cl_updatePosCyclicWallsKernel, 0, sizeof(cl_mem), &cl_posBuff);
  clSetKernelArg(cl_updatePosCyclicWallsKernel, 1, sizeof(cl_mem), &cl_velBuff);
  ///

  return true;
}

bool Boids::acquireGLBuffers(const std::vector<cl_mem>& GLBuffers)
{
  if (!m_init)
    return false;

  auto cl_queue = m_clContext->cl_queue;

  for (const auto& GLBuffer : GLBuffers)
  {
    cl_int err = clEnqueueAcquireGLObjects(cl_queue, 1, &GLBuffer, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
      printf("error when acquiring GL buffer");
  }

  return true;
}

bool Boids::releaseGLBuffers(const std::vector<cl_mem>& GLBuffers)
{
  if (!m_init)
    return false;

  auto cl_queue = m_clContext->cl_queue;

  for (const auto& GLBuffer : GLBuffers)
  {
    cl_int err = clEnqueueReleaseGLObjects(cl_queue, 1, &GLBuffer, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
      printf("error when releasing GL buffer");
  }

  clFinish(cl_queue);

  return true;
}

void Boids::runKernel(cl_kernel kernel, double* profilingTimeMs)
{
  if (!m_init)
    return;

  auto cl_queue = m_clContext->cl_queue;

  cl_event event;

  size_t numWorkItems = (m_numEntities > 10) ? m_numEntities - (m_numEntities % 10) : m_numEntities;
  clEnqueueNDRangeKernel(cl_queue, kernel, 1, NULL, &numWorkItems, NULL, 0, NULL, &event);

  //if (m_kernelProfilingEnabled)
  {
    cl_int err;

    err = clFlush(cl_queue);
    if (err != CL_SUCCESS)
      printf("error when flushing opencl run");

    err = clFinish(cl_queue);
    if (err != CL_SUCCESS)
      printf("error when finishing opencl run");

    cl_ulong start = 0, end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    //the resolution of the events is 1e-09 sec
    *profilingTimeMs = (double)((cl_double)(end - start) * (1e-06));
  }
}