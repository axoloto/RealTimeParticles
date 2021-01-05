#include "Boids.hpp"
#include "windows.h" // WIP
#include <ctime>
#include <iostream>
#include <spdlog/spdlog.h>

using namespace Core;

#define PROGRAM_BOIDS "boids"

#define KERNEL_RANDOM_POS "randPosVerts"
#define KERNEL_BOIDS_RULES "applyBoidsRules"
#define KERNEL_UPDATE_POS_BOUNCING "updatePosVertsWithBouncingWalls"
#define KERNEL_UPDATE_POS_CYCLIC "updatePosVertsWithCyclicWalls"
#define KERNEL_UPDATE_VEL "updateVelVerts"
#define KERNEL_FLUSH_GRID_CELLS "flushGridCells"
#define KERNEL_FILL_GRID_CELLS "fillGridCells"
#define KERNEL_COLOR "colorVerts"

Boids::Boids(size_t numEntities, size_t gridRes,
    unsigned int pointCloudCoordVBO,
    unsigned int pointCloudColorVBO,
    unsigned int gridDetectorVBO)
    : Physics(numEntities, gridRes)
    , m_scaleAlignment(1.6f)
    , m_scaleCohesion(0.7f)
    , m_scaleSeparation(1.6f)
    , m_activeTargets(false)
    , m_activeAlignment(true)
    , m_activeSeparation(true)
    , m_activeCohesion(true)
    , m_target({ 0.0f, 0.0f, 0.0f })
{
  if (m_clContext.init())
  {
    // WIP, hardcoded Path
    m_clContext.createProgram(PROGRAM_BOIDS,
        "C:\\Dev_perso\\boids\\physics\\ocl\\kernels\\boids.cl",
        "-DEFFECT_RADIUS_SQUARED=1000 -DMAX_STEERING=0.5f -DMAX_VELOCITY=5.0f -DABS_WALL_POS=250.0f -DFLOAT_EPSILON=0.0001f");

    createBuffers(pointCloudCoordVBO, pointCloudColorVBO, gridDetectorVBO);

    createKernels();

    m_init = true;

    reset();
  }
}

bool Boids::createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO, unsigned int gridDetectorVBO)
{
  m_clContext.createGLBuffer("boidsColor", pointCloudColorVBO, CL_MEM_WRITE_ONLY);
  m_clContext.createGLBuffer("boidsPos", pointCloudCoordVBO, CL_MEM_READ_WRITE);
  m_clContext.createGLBuffer("gridDetector", gridDetectorVBO, CL_MEM_READ_WRITE);

  size_t boidsBufferSize = 4 * NUM_MAX_ENTITIES * sizeof(float);

  m_clContext.createBuffer("boidsVel", boidsBufferSize, CL_MEM_READ_WRITE);
  m_clContext.createBuffer("boidsAcc", boidsBufferSize, CL_MEM_READ_WRITE);
  m_clContext.createBuffer("boidsParams", sizeof(m_boidsParams), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);
  m_clContext.createBuffer("gridParams", sizeof(m_gridParams), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);

  return true;
}

bool Boids::createKernels()
{
  m_clContext.createKernel(PROGRAM_BOIDS, KERNEL_COLOR, { "boidsColor" });
  m_clContext.createKernel(PROGRAM_BOIDS, KERNEL_RANDOM_POS, { "boidsPos", "boidsVel", "boidsParams" });
  m_clContext.createKernel(PROGRAM_BOIDS, KERNEL_BOIDS_RULES, { "boidsPos", "boidsVel", "boidsAcc", "boidsParams" });
  m_clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_VEL, { "boidsVel", "boidsAcc", "boidsParams" });
  m_clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_POS_BOUNCING, { "boidsPos", "boidsVel" });
  m_clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_POS_CYCLIC, { "boidsPos", "boidsVel" });
  m_clContext.createKernel(PROGRAM_BOIDS, KERNEL_FLUSH_GRID_CELLS, { "gridDetector" });
  m_clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_GRID_CELLS, { "boidsPos", "gridDetector", "gridParams" });

  return true;
}

void Boids::updateGridParamsInKernel()
{
  m_gridParams.gridRes = (cl_uint)m_gridRes;
  m_gridParams.numCells = (cl_uint)(m_gridRes * m_gridRes * m_gridRes);

  m_clContext.mapAndSendBufferToDevice("gridParams", &m_gridParams, sizeof(m_gridParams));
}

void Boids::updateBoidsParamsInKernel()
{
  m_boidsParams.dims = (m_dimension == Dimension::dim2D) ? 2.0f : 3.0f;
  m_boidsParams.velocity = m_velocity;
  m_boidsParams.scaleCohesion = m_activeCohesion ? m_scaleCohesion : 0.0f;
  m_boidsParams.scaleAlignment = m_activeAlignment ? m_scaleAlignment : 0.0f;
  m_boidsParams.scaleSeparation = m_activeSeparation ? m_scaleSeparation : 0.0f;
  m_boidsParams.activeTarget = m_activeTargets ? 1 : 0;

  m_clContext.mapAndSendBufferToDevice("boidsParams", &m_boidsParams, sizeof(m_boidsParams));
}

void Boids::reset()
{
  if (!m_init)
    return;

  updateGridParamsInKernel();

  updateBoidsParamsInKernel();

  m_clContext.acquireGLBuffers({ "boidsColor", "boidsPos", "gridDetector" });
  m_clContext.runKernel(KERNEL_COLOR, m_numEntities);

  m_clContext.runKernel(KERNEL_RANDOM_POS, m_numEntities);

  m_clContext.runKernel(KERNEL_FLUSH_GRID_CELLS, m_gridParams.numCells);

  m_clContext.runKernel(KERNEL_FILL_GRID_CELLS, m_numEntities);

  m_clContext.releaseGLBuffers({ "boidsColor", "boidsPos", "gridDetector" });
}

void Boids::update()
{
  if (!m_init || m_pause)
    return;

  m_clContext.acquireGLBuffers({ "boidsPos", "gridDetector" });
  m_clContext.runKernel(KERNEL_BOIDS_RULES, m_numEntities);
  m_clContext.runKernel(KERNEL_UPDATE_VEL, m_numEntities);

  if (m_boundary == Boundary::CyclicWall)
    m_clContext.runKernel(KERNEL_UPDATE_POS_CYCLIC, m_numEntities);
  else if (m_boundary == Boundary::BouncingWall)
    m_clContext.runKernel(KERNEL_UPDATE_POS_BOUNCING, m_numEntities);

  m_clContext.runKernel(KERNEL_FLUSH_GRID_CELLS, m_gridParams.numCells);
  m_clContext.runKernel(KERNEL_FILL_GRID_CELLS, m_numEntities);
  m_clContext.releaseGLBuffers({ "boidsPos", "gridDetector" });
}