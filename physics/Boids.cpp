#include "Boids.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <spdlog/spdlog.h>
#include <sstream>

using namespace Core;

#define PROGRAM_BOIDS "boids"

#define KERNEL_RANDOM_POS "randPosVerts"
#define KERNEL_BOIDS_RULES "applyBoidsRules"
#define KERNEL_UPDATE_POS_BOUNCING "updatePosVertsWithBouncingWalls"
#define KERNEL_UPDATE_POS_CYCLIC "updatePosVertsWithCyclicWalls"
#define KERNEL_UPDATE_VEL "updateVelVerts"
#define KERNEL_FLUSH_GRID_DETECTOR "flushGridDetector"
#define KERNEL_FILL_GRID_DETECTOR "fillGridDetector"
#define KERNEL_COLOR "colorVerts"
#define KERNEL_FLUSH_CELL_ID "flushCellIDs"
#define KERNEL_FILL_CELL_ID "fillCellIDs"
#define KERNEL_FLUSH_START_END_CELL "flushStartEndCell"
#define KERNEL_FILL_START_CELL "fillStartCell"
#define KERNEL_FILL_END_CELL "fillEndCell"
#define KERNEL_FILL_TEXT "fillBoidsTexture"
#define KERNEL_BOIDS_RULES_GRID "applyBoidsRulesWithGrid"
#define KERNEL_BOIDS_RULES_GRID_TEXT "applyBoidsRulesWithGridAndTex"

Boids::Boids(size_t numEntities, size_t boxSize, size_t gridRes,
    unsigned int pointCloudCoordVBO,
    unsigned int pointCloudColorVBO,
    unsigned int gridDetectorVBO)
    : Physics(numEntities, boxSize, gridRes)
    , m_scaleAlignment(1.6f)
    , m_scaleCohesion(0.7f)
    , m_scaleSeparation(1.6f)
    , m_activeTargets(false)
    , m_activeAlignment(true)
    , m_activeSeparation(true)
    , m_activeCohesion(true)
    , m_target({ 0.0f, 0.0f, 0.0f })
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
  clBuildOptions << "-DEFFECT_RADIUS_SQUARED=1000 ";
  clBuildOptions << " -DMAX_STEERING=0.5f ";
  clBuildOptions << " -DMAX_VELOCITY=5.0f ";
  clBuildOptions << " -DABS_WALL_POS=" << std::fixed << std::setprecision(2)
                 << std::setfill('0') << m_boxSize / 2.0f << "f";
  clBuildOptions << " -DFLOAT_EPSILON=0.01f";
  clBuildOptions << " -DGRID_RES=" << m_gridRes;
  clBuildOptions << " -DGRID_NUM_CELLS=" << (m_gridRes * m_gridRes * m_gridRes);

  // WIP, hardcoded Path
  clContext.createProgram(PROGRAM_BOIDS,
      "C:\\Dev_perso\\boids\\physics\\ocl\\kernels\\boids.cl", clBuildOptions.str());

  return true;
}

bool Boids::createBuffers(unsigned int pointCloudCoordVBO, unsigned int pointCloudColorVBO, unsigned int gridDetectorVBO) const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createGLBuffer("boidsColor", pointCloudColorVBO, CL_MEM_WRITE_ONLY);
  clContext.createGLBuffer("boidsPos", pointCloudCoordVBO, CL_MEM_READ_WRITE);
  clContext.createGLBuffer("gridDetector", gridDetectorVBO, CL_MEM_READ_WRITE);

  clContext.createBuffer("boidsVel", 4 * NUM_MAX_ENTITIES * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("boidsAcc", 4 * NUM_MAX_ENTITIES * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("boidsCellIDs", NUM_MAX_ENTITIES * sizeof(unsigned int), CL_MEM_READ_WRITE);

  clContext.createBuffer("startEndBoidsIndices", 2 * m_gridRes * m_gridRes * m_gridRes * sizeof(unsigned int), CL_MEM_READ_WRITE);

  CL::imageSpecs imageSpecs { CL_RGBA, CL_FLOAT, 200, m_gridRes * m_gridRes * m_gridRes };
  clContext.createImage2D("boidsPosText", imageSpecs, CL_MEM_READ_WRITE);
  clContext.createImage2D("boidsVelText", imageSpecs, CL_MEM_READ_WRITE);

  return true;
}

bool Boids::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();

  // Init only
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_COLOR, { "boidsColor" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_RANDOM_POS, { "boidsPos", "boidsVel" });

  // For rendering purpose only
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FLUSH_GRID_DETECTOR, { "gridDetector" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_GRID_DETECTOR, { "boidsPos", "gridDetector" });

  // Boids Physics
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_VEL, { "boidsVel", "boidsAcc" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_POS_BOUNCING, { "boidsPos", "boidsVel" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_POS_CYCLIC, { "boidsPos", "boidsVel" });

  // Radix Sort based on 3D grid
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FLUSH_CELL_ID, { "boidsCellIDs" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_CELL_ID, { "boidsPos", "boidsCellIDs" });

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FLUSH_START_END_CELL, { "startEndBoidsIndices" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_START_CELL, { "boidsCellIDs", "startEndBoidsIndices" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_END_CELL, { "boidsCellIDs", "startEndBoidsIndices" });

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_TEXT, { "startEndBoidsIndices", "boidsPos", "boidsPosText" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_BOIDS_RULES_GRID_TEXT, { "boidsPos", "boidsVel", "boidsPosText", "boidsVelText", "boidsAcc" });

  return true;
}

void Boids::updateBoidsParamsInKernel()
{
  CL::Context& clContext = CL::Context::Get();

  float dim = (m_dimension == Dimension::dim2D) ? 2.0f : 3.0f;
  clContext.setKernelArg(KERNEL_RANDOM_POS, 2, sizeof(float), &dim);

  float velVal = m_velocity;
  clContext.setKernelArg(KERNEL_UPDATE_VEL, 2, sizeof(float), &velVal);

  std::array<float, 8> boidsParams;
  boidsParams[0] = m_velocity;
  boidsParams[1] = m_activeCohesion ? m_scaleCohesion : 0.0f;
  boidsParams[2] = m_activeAlignment ? m_scaleAlignment : 0.0f;
  boidsParams[3] = m_activeSeparation ? m_scaleSeparation : 0.0f;
  boidsParams[4] = m_activeTargets ? 1.0f : 0.0f;
  clContext.setKernelArg(KERNEL_BOIDS_RULES_GRID_TEXT, 5, sizeof(boidsParams), &boidsParams);
}

void Boids::reset()
{
  if (!m_init)
    return;

  updateBoidsParamsInKernel();

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "boidsColor", "boidsPos", "gridDetector" });
  clContext.runKernel(KERNEL_COLOR, m_numEntities);
  clContext.runKernel(KERNEL_RANDOM_POS, m_numEntities);
  clContext.runKernel(KERNEL_FLUSH_GRID_DETECTOR, m_gridRes * m_gridRes * m_gridRes);
  clContext.runKernel(KERNEL_FILL_GRID_DETECTOR, m_numEntities);
  clContext.releaseGLBuffers({ "boidsColor", "boidsPos", "gridDetector" });
}

void Boids::update()
{
  if (!m_init || m_pause)
    return;

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "boidsPos", "gridDetector" });
  clContext.runKernel(KERNEL_FLUSH_CELL_ID, NUM_MAX_ENTITIES);
  clContext.runKernel(KERNEL_FILL_CELL_ID, m_numEntities);

  m_radixSort.sort("boidsCellIDs", { "boidsPos", "boidsVel", "boidsAcc" });

  clContext.runKernel(KERNEL_FLUSH_START_END_CELL, m_gridRes * m_gridRes * m_gridRes);
  clContext.runKernel(KERNEL_FILL_START_CELL, m_numEntities);
  clContext.runKernel(KERNEL_FILL_END_CELL, m_numEntities);

  clContext.setKernelArg(KERNEL_FILL_TEXT, 1, "boidsPos");
  clContext.setKernelArg(KERNEL_FILL_TEXT, 2, "boidsPosText");
  clContext.runKernel(KERNEL_FILL_TEXT, m_gridRes * m_gridRes * m_gridRes * 200, 200);

  clContext.setKernelArg(KERNEL_FILL_TEXT, 1, "boidsVel");
  clContext.setKernelArg(KERNEL_FILL_TEXT, 2, "boidsVelText");
  clContext.runKernel(KERNEL_FILL_TEXT, m_gridRes * m_gridRes * m_gridRes * 200, 200);

  clContext.runKernel(KERNEL_BOIDS_RULES_GRID_TEXT, m_numEntities);
  clContext.runKernel(KERNEL_UPDATE_VEL, m_numEntities);

  if (m_boundary == Boundary::CyclicWall)
    clContext.runKernel(KERNEL_UPDATE_POS_CYCLIC, m_numEntities);
  else if (m_boundary == Boundary::BouncingWall)
    clContext.runKernel(KERNEL_UPDATE_POS_BOUNCING, m_numEntities);

  clContext.runKernel(KERNEL_FLUSH_GRID_DETECTOR, m_gridRes * m_gridRes * m_gridRes);
  clContext.runKernel(KERNEL_FILL_GRID_DETECTOR, m_numEntities);

  clContext.releaseGLBuffers({ "boidsPos", "gridDetector" });
}