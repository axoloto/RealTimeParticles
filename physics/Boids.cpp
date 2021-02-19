#include "Boids.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <spdlog/spdlog.h>
#include <sstream>

using namespace Core;

#define PROGRAM_BOIDS "boids"

#define KERNEL_RANDOM_POS "randPosVerts"
#define KERNEL_UPDATE_POS_BOUNCING "updatePosWithBouncingWalls"
#define KERNEL_UPDATE_POS_CYCLIC "updatePosWithCyclicWalls"
#define KERNEL_UPDATE_VEL "updateVel"
#define KERNEL_FLUSH_GRID_DETECTOR "flushGridDetector"
#define KERNEL_FILL_GRID_DETECTOR "fillGridDetector"
#define KERNEL_COLOR "colorVerts"
#define KERNEL_FLUSH_CELL_ID "flushCellIDs"
#define KERNEL_FILL_CELL_ID "fillCellIDs"
#define KERNEL_FLUSH_START_END_CELL "flushStartEndCell"
#define KERNEL_FILL_START_CELL "fillStartCell"
#define KERNEL_FILL_END_CELL "fillEndCell"
#define KERNEL_ADJUST_END_CELL "adjustEndCell"
#define KERNEL_FILL_TEXT "fillBoidsTexture"
#define KERNEL_BOIDS_RULES_GRID "applyBoidsRulesWithGrid"
#define KERNEL_ADD_TARGET_RULE "addTargetRule"

Boids::Boids(size_t numEntities, size_t boxSize, size_t gridRes, float velocity,
    unsigned int pointCloudCoordVBO,
    unsigned int pointCloudColorVBO,
    unsigned int gridDetectorVBO)
    : Physics(numEntities, boxSize, gridRes, velocity)
    , m_scaleAlignment(1.6f)
    , m_scaleCohesion(0.7f)
    , m_scaleSeparation(1.6f)
    , m_activeTargets(false)
    , m_activeAlignment(true)
    , m_activeSeparation(true)
    , m_activeCohesion(true)
    , m_target({ 0.0f, 0.0f, 0.0f })
    , m_targetRadiusEffect(10000.0f)
    , m_targetSign(1)
    , m_maxNbPartsInCell(150)
    , m_radixSort(NUM_MAX_ENTITIES)
{
  createProgram();

  createBuffers(pointCloudCoordVBO, pointCloudColorVBO, gridDetectorVBO);

  createKernels();

  m_init = true;

  reset();
}

bool Boids::createProgram() const
{
  CL::Context& clContext = CL::Context::Get();

  std::ostringstream clBuildOptions;
  clBuildOptions << "-DEFFECT_RADIUS_SQUARED=1500 ";
  clBuildOptions << " -DMAX_STEERING=0.5f ";
  clBuildOptions << " -DABS_WALL_POS=" << std::fixed << std::setprecision(2)
                 << std::setfill('0') << m_boxSize / 2.0f << "f";
  clBuildOptions << " -DFLOAT_EPSILON=0.01f";
  clBuildOptions << " -DGRID_RES=" << m_gridRes;
  clBuildOptions << " -DGRID_NUM_CELLS=" << (m_gridRes * m_gridRes * m_gridRes);
  clBuildOptions << " -DNUM_MAX_PARTS_IN_CELL=" << m_maxNbPartsInCell;

  clContext.createProgram(PROGRAM_BOIDS, ".\\physics\\ocl\\kernels\\boids.cl", clBuildOptions.str());

  return true;
}

bool Boids::createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO, unsigned int gridDetectorVBO) const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createGLBuffer("p_Color", pointCloudColorVBO, CL_MEM_WRITE_ONLY);
  clContext.createGLBuffer("p_Pos", pointCloudCoordVBO, CL_MEM_READ_WRITE);
  clContext.createGLBuffer("c_partDetector", gridDetectorVBO, CL_MEM_READ_WRITE);

  clContext.createBuffer("p_Vel", 4 * NUM_MAX_ENTITIES * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_Acc", 4 * NUM_MAX_ENTITIES * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_CellID", NUM_MAX_ENTITIES * sizeof(unsigned int), CL_MEM_READ_WRITE);

  clContext.createBuffer("c_startEndPartID", 2 * m_gridRes * m_gridRes * m_gridRes * sizeof(unsigned int), CL_MEM_READ_WRITE);

  return true;
}

bool Boids::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();

  // Init only
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_COLOR, { "p_Pos", "p_Color" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_RANDOM_POS, { "p_Pos", "p_Vel" });

  // For rendering purpose only
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FLUSH_GRID_DETECTOR, { "c_partDetector" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_GRID_DETECTOR, { "p_Pos", "c_partDetector" });

  // Boids Physics
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_VEL, { "p_Acc", "", "", "p_Vel" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_POS_BOUNCING, { "p_Vel", "", "p_Pos" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_POS_CYCLIC, { "p_Vel", "", "p_Pos" });

  // Radix Sort based on 3D grid
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FLUSH_CELL_ID, { "p_CellID" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_CELL_ID, { "p_Pos", "p_CellID" });

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FLUSH_START_END_CELL, { "c_startEndPartID" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_START_CELL, { "p_CellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_END_CELL, { "p_CellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_ADJUST_END_CELL, { "c_startEndPartID" });

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_BOIDS_RULES_GRID, { "p_Pos", "p_Vel", "c_startEndPartID", "", "p_Acc" });

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_ADD_TARGET_RULE, { "p_Pos", "", "", "p_Acc", "p_Acc" });

  return true;
}

void Boids::updateBoidsParamsInKernel()
{
  CL::Context& clContext = CL::Context::Get();

  float dim = (m_dimension == Dimension::dim2D) ? 2.0f : 3.0f;
  clContext.setKernelArg(KERNEL_RANDOM_POS, 2, sizeof(float), &dim);

  float vel = m_velocity;
  clContext.setKernelArg(KERNEL_UPDATE_VEL, 2, sizeof(float), &vel);

  std::array<float, 8> boidsParams;
  boidsParams[0] = m_velocity;
  boidsParams[1] = m_activeCohesion ? m_scaleCohesion : 0.0f;
  boidsParams[2] = m_activeAlignment ? m_scaleAlignment : 0.0f;
  boidsParams[3] = m_activeSeparation ? m_scaleSeparation : 0.0f;
  boidsParams[4] = m_activeTargets ? 1.0f : 0.0f;
  clContext.setKernelArg(KERNEL_BOIDS_RULES_GRID, 3, sizeof(boidsParams), &boidsParams);

  std::array<float, 4> targetPos = { m_target.x, m_target.y, m_target.z, 0.0f };
  clContext.setKernelArg(KERNEL_ADD_TARGET_RULE, 1, sizeof(float) * 4, &targetPos);
  clContext.setKernelArg(KERNEL_ADD_TARGET_RULE, 2, sizeof(float), &m_targetRadiusEffect);
  clContext.setKernelArg(KERNEL_ADD_TARGET_RULE, 3, sizeof(int), &m_targetSign);
}

void Boids::reset()
{
  if (!m_init)
    return;

  updateBoidsParamsInKernel();

  m_time = clock::now();
  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_Color", "p_Pos", "c_partDetector" });
  clContext.runKernel(KERNEL_RANDOM_POS, m_numEntities);
  clContext.runKernel(KERNEL_COLOR, m_numEntities);
  clContext.runKernel(KERNEL_FLUSH_GRID_DETECTOR, m_gridRes * m_gridRes * m_gridRes);
  clContext.runKernel(KERNEL_FILL_GRID_DETECTOR, m_numEntities);
  clContext.releaseGLBuffers({ "p_Color", "p_Pos", "c_partDetector" });
}

void Boids::update()
{
  if (!m_init || m_pause)
    return;

  CL::Context& clContext = CL::Context::Get();

  auto currentTime = clock::now();
  float timeStep = (float)(std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - m_time).count()) / 16.0f;
  m_time = currentTime;

  clContext.acquireGLBuffers({ "p_Color", "p_Pos", "c_partDetector" });
  clContext.runKernel(KERNEL_FLUSH_CELL_ID, NUM_MAX_ENTITIES);
  clContext.runKernel(KERNEL_FILL_CELL_ID, m_numEntities);

  m_radixSort.sort("p_CellID", { "p_Color", "p_Pos", "p_Vel", "p_Acc" });

  clContext.runKernel(KERNEL_FLUSH_START_END_CELL, m_gridRes * m_gridRes * m_gridRes);
  clContext.runKernel(KERNEL_FILL_START_CELL, m_numEntities);
  clContext.runKernel(KERNEL_FILL_END_CELL, m_numEntities);
  clContext.runKernel(KERNEL_ADJUST_END_CELL, m_gridRes * m_gridRes * m_gridRes);

  clContext.runKernel(KERNEL_BOIDS_RULES_GRID, m_numEntities);

  if (m_activeTargets)
  {
    clContext.runKernel(KERNEL_ADD_TARGET_RULE, m_numEntities);
  }

  clContext.setKernelArg(KERNEL_UPDATE_VEL, 1, sizeof(float), &timeStep);
  clContext.runKernel(KERNEL_UPDATE_VEL, m_numEntities);

  switch (m_boundary)
  {
  case Boundary::CyclicWall:
    clContext.setKernelArg(KERNEL_UPDATE_POS_CYCLIC, 1, sizeof(float), &timeStep);
    clContext.runKernel(KERNEL_UPDATE_POS_CYCLIC, m_numEntities);
    break;
  case Boundary::BouncingWall:
    clContext.setKernelArg(KERNEL_UPDATE_POS_BOUNCING, 1, sizeof(float), &timeStep);
    clContext.runKernel(KERNEL_UPDATE_POS_BOUNCING, m_numEntities);
    break;
  }

  clContext.runKernel(KERNEL_FLUSH_GRID_DETECTOR, m_gridRes * m_gridRes * m_gridRes);
  clContext.runKernel(KERNEL_FILL_GRID_DETECTOR, m_numEntities);

  clContext.releaseGLBuffers({ "p_Color", "p_Pos", "c_partDetector" });
}