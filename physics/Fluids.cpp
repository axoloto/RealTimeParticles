#include "Fluids.hpp"
#include "Logging.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <array>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <chrono>
#include <thread>

using namespace Physics;

#define PROGRAM_FLUIDS "fluids"

// utils.cl
#define KERNEL_INFINITE_POS "infPosVerts"
#define KERNEL_RESET_CAMERA_DIST "resetCameraDist"
#define KERNEL_FILL_CAMERA_DIST "fillCameraDist"

// grid.cl
#define KERNEL_RESET_GRID_DETECTOR "resetGridDetector"
#define KERNEL_FILL_GRID_DETECTOR "fillGridDetector"
#define KERNEL_RESET_CELL_ID "resetCellIDs"
#define KERNEL_FILL_CELL_ID "fillCellIDs"
#define KERNEL_RESET_START_END_CELL "resetStartEndCell"
#define KERNEL_FILL_START_CELL "fillStartCell"
#define KERNEL_FILL_END_CELL "fillEndCell"
#define KERNEL_ADJUST_END_CELL "adjustEndCell"

// fluids.cl
#define KERNEL_RANDOM_POS "randPosVertsFluid"
#define KERNEL_UPDATE_POS_BOUNCING "updatePosWithBouncingWalls"
#define KERNEL_PREDICT_POS "predictPosition"
#define KERNEL_APPLY_BOUNDARY "applyBoundaryCondition"
#define KERNEL_DENSITY "computeDensity"
#define KERNEL_CONSTRAINT_FACTOR "computeConstraintFactor"
#define KERNEL_CONSTRAINT_CORRECTION "computeConstraintCorrection"
#define KERNEL_CORRECT_POS "correctPosition"
#define KERNEL_UPDATE_POS "updatePosition"
#define KERNEL_UPDATE_VEL "updateVel"

#define MAX_NB_JACOBI_ITERS 3

Fluids::Fluids(ModelParams params)
    : Model(params)
    , m_scaleAlignment(1.6f)
    , m_scaleCohesion(1.45f)
    , m_scaleSeparation(1.6f)
    , m_activeAlignment(true)
    , m_activeSeparation(true)
    , m_activeCohesion(true)
    , m_simplifiedMode(true)
    , m_maxNbPartsInCell(1000)
    , m_radixSort(params.maxNbParticles)
    , m_target(std::make_unique<Target>(params.boxSize))
{
  createProgram();

  createBuffers();

  createKernels();

  m_init = true;

  reset();
}

bool Fluids::createProgram() const
{
  CL::Context& clContext = CL::Context::Get();

  float effectRadius = ((float)m_boxSize) / m_gridRes;

  std::ostringstream clBuildOptions;
  clBuildOptions << "-DEFFECT_RADIUS=" << Utils::FloatToStr(effectRadius);
  clBuildOptions << " -DABS_WALL_POS=" << Utils::FloatToStr(m_boxSize / 2.0f);
  clBuildOptions << " -DGRID_RES=" << m_gridRes;
  clBuildOptions << " -DGRID_CELL_SIZE=" << Utils::FloatToStr((float)m_boxSize / m_gridRes);
  clBuildOptions << " -DGRID_NUM_CELLS=" << m_nbCells;
  clBuildOptions << " -DNUM_MAX_PARTS_IN_CELL=" << m_maxNbPartsInCell;
  clBuildOptions << " -DPOLY6_COEFF=" << Utils::FloatToStr(315.0f / (64.0f * Math::PI_F * std::powf(effectRadius, 9)));
  // clBuildOptions << " -DPOLY6_COEFF=" << Utils::FloatToStr(4.0f / (Math::PI_F * std::powf(effectRadius, 8)));// Shallow water
  clBuildOptions << " -DSPIKY_COEFF=" << Utils::FloatToStr(15.0f / (Math::PI_F * std::powf(effectRadius, 6)));
  // clBuildOptions << " -DSPIKY_COEFF=" << Utils::FloatToStr(4.0f / (Math::PI_F * std::powf(effectRadius, 8)));// Shallow water
  clBuildOptions << " -DREST_DENSITY=" << Utils::FloatToStr(10.0f); // TODO
  clBuildOptions << " -DRELAX_CFM=" << Utils::FloatToStr(600.0f); // TODO

  LOG_INFO(clBuildOptions.str());
  clContext.createProgram(PROGRAM_FLUIDS, std::vector<std::string>({ "fluids.cl", "utils.cl", "grid.cl" }), clBuildOptions.str());

  return true;
}

bool Fluids::createBuffers() const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createGLBuffer("u_cameraPos", m_cameraVBO, CL_MEM_READ_ONLY);
  clContext.createGLBuffer("p_pos", m_particleVBO, CL_MEM_READ_WRITE);
  clContext.createGLBuffer("c_partDetector", m_gridVBO, CL_MEM_READ_WRITE);

  clContext.createBuffer("p_density", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_predPos", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_corrPos", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_constFactor", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_vel", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cellID", m_maxNbParticles * sizeof(unsigned int), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cameraDist", m_maxNbParticles * sizeof(unsigned int), CL_MEM_READ_WRITE);

  clContext.createBuffer("c_startEndPartID", 2 * m_nbCells * sizeof(unsigned int), CL_MEM_READ_WRITE);

  return true;
}

bool Fluids::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();

  // Init only
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_INFINITE_POS, { "p_pos" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RANDOM_POS, { "p_pos", "p_vel" });

  // For rendering purpose only
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RESET_GRID_DETECTOR, { "c_partDetector" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_GRID_DETECTOR, { "p_pos", "c_partDetector" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RESET_CAMERA_DIST, { "p_cameraDist" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_CAMERA_DIST, { "p_pos", "u_cameraPos", "p_cameraDist" });

  // Radix Sort based on 3D grid, using predicted positions, not corrected ones
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RESET_CELL_ID, { "p_cellID" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_CELL_ID, { "p_predPos", "p_cellID" });

  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RESET_START_END_CELL, { "c_startEndPartID" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_START_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_END_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_ADJUST_END_CELL, { "c_startEndPartID" });

  // Position Based Fluids
  /// Position prediction
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_PREDICT_POS, { "p_pos", "p_vel", "", "", "p_predPos" });
  /// Boundary conditions
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_APPLY_BOUNDARY, { "p_predPos" });
  /// Jacobi solver to correct position
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_DENSITY, { "p_predPos", "c_startEndPartID", "p_density" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_CONSTRAINT_FACTOR, { "p_predPos", "p_density", "c_startEndPartID", "p_constFactor" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_CONSTRAINT_CORRECTION, { "p_constFactor", "c_startEndPartID", "p_predPos", "p_corrPos" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_CORRECT_POS, { "p_corrPos", "p_predPos" });
  /// Velocity and position update
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_UPDATE_VEL, { "p_predPos", "p_pos", "", "p_vel" });
  //
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_UPDATE_POS_BOUNCING, { "p_predPos", "p_pos" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_UPDATE_POS, { "p_predPos", "p_pos" });

  return true;
}

void Fluids::updateFluidsParamsInKernel()
{
  CL::Context& clContext = CL::Context::Get();

  float dim = (m_dimension == Dimension::dim2D) ? 2.0f : 3.0f;
  clContext.setKernelArg(KERNEL_RANDOM_POS, 2, sizeof(float), &dim);

  clContext.setKernelArg(KERNEL_PREDICT_POS, 3, sizeof(float), &m_velocity);

  float timeStep = 0.008;
  clContext.setKernelArg(KERNEL_PREDICT_POS, 2, sizeof(float), &timeStep);

  clContext.setKernelArg(KERNEL_UPDATE_VEL, 2, sizeof(float), &timeStep);

  /*
  std::array<float, 8> boidsParams;
  boidsParams[0] = m_velocity;
  boidsParams[1] = m_activeCohesion ? m_scaleCohesion : 0.0f;
  boidsParams[2] = m_activeAlignment ? m_scaleAlignment : 0.0f;
  boidsParams[3] = m_activeSeparation ? m_scaleSeparation : 0.0f;
  boidsParams[4] = isTargetActivated() ? 1.0f : 0.0f;
  clContext.setKernelArg(KERNEL_BOIDS_RULES_GRID_2D, 3, sizeof(boidsParams), &boidsParams);
  clContext.setKernelArg(KERNEL_BOIDS_RULES_GRID_3D, 3, sizeof(boidsParams), &boidsParams);
 */
}

void Fluids::reset()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_pos", "c_partDetector" });

  updateFluidsParamsInKernel();

  // WIP
  //clContext.runKernel(KERNEL_INFINITE_POS, m_maxNbParticles);
  //clContext.runKernel(KERNEL_RANDOM_POS, m_currNbParticles);

  // Dam setup
  float inf = std::numeric_limits<float>::infinity();
  std::vector<std::array<float, 4>> pos(m_maxNbParticles, std::array<float, 4>({ inf, inf, inf, 0.0f }));

  int i = 0;
  float effectRadius = ((float)m_boxSize) / m_gridRes;
  float gridSpacing = 0.5f * effectRadius;
  Math::float3 startFluidPos = { 0.0f, 0.0f, 0.0f };
  for (int ix = 0; ix < 64; ++ix)
  {
    for (int iy = 0; iy < 32; ++iy)
    {
      for (int iz = 0; iz < 32; ++iz)
      {
        pos[i++] = { ix * gridSpacing - startFluidPos.x,
          iy * gridSpacing - startFluidPos.y,
          iz * gridSpacing - startFluidPos.z,
          0.0f };
      }
    }
  }
  clContext.loadBufferFromHost("p_pos", 0, 4 * sizeof(float) * pos.size(), pos.data());

  std::vector<std::array<float, 4>> vel(m_maxNbParticles, std::array<float, 4>({ 0.0f, 0.0f, 0.0f, 0.0f }));
  clContext.loadBufferFromHost("p_vel", 0, 4 * sizeof(float) * vel.size(), vel.data());

  clContext.runKernel(KERNEL_RESET_GRID_DETECTOR, m_nbCells);
  clContext.runKernel(KERNEL_FILL_GRID_DETECTOR, m_currNbParticles);

  clContext.runKernel(KERNEL_RESET_CELL_ID, m_maxNbParticles);
  clContext.runKernel(KERNEL_RESET_CAMERA_DIST, m_maxNbParticles);

  clContext.releaseGLBuffers({ "p_pos", "c_partDetector" });
}

void Fluids::update()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_pos", "c_partDetector", "u_cameraPos" });

  if (!m_pause)
  {
    // Predicting velocity and position
    clContext.runKernel(KERNEL_PREDICT_POS, m_currNbParticles);

    // NNS - spatial partitioning
    clContext.runKernel(KERNEL_FILL_CELL_ID, m_currNbParticles);

    m_radixSort.sort("p_cellID", { "p_pos", "p_vel", "p_predPos" });

    clContext.runKernel(KERNEL_RESET_START_END_CELL, m_nbCells);
    clContext.runKernel(KERNEL_FILL_START_CELL, m_currNbParticles);
    clContext.runKernel(KERNEL_FILL_END_CELL, m_currNbParticles);

    if (m_simplifiedMode)
      clContext.runKernel(KERNEL_ADJUST_END_CELL, m_nbCells);

    // Correcting positions to fit constraints
    for (int iter = 0; iter < MAX_NB_JACOBI_ITERS; ++iter)
    {
      // Clamping to boundary
      clContext.runKernel(KERNEL_APPLY_BOUNDARY, m_currNbParticles);
      // Computing density using SPH method
      clContext.runKernel(KERNEL_DENSITY, m_currNbParticles);
      // Computing constraint factor Lambda
      clContext.runKernel(KERNEL_CONSTRAINT_FACTOR, m_currNbParticles);
      // Computing position correction
      clContext.runKernel(KERNEL_CONSTRAINT_CORRECTION, m_currNbParticles);
      // Correcting predicted position
      clContext.runKernel(KERNEL_CORRECT_POS, m_currNbParticles);
    }

    // Updating velocity and position
    clContext.runKernel(KERNEL_UPDATE_VEL, m_currNbParticles);
    clContext.runKernel(KERNEL_UPDATE_POS, m_currNbParticles);

    // Rendering purpose
    clContext.runKernel(KERNEL_RESET_GRID_DETECTOR, m_nbCells);
    clContext.runKernel(KERNEL_FILL_GRID_DETECTOR, m_currNbParticles);
  }

  // Rendering purpose
  clContext.runKernel(KERNEL_FILL_CAMERA_DIST, m_currNbParticles);

  m_radixSort.sort("p_cameraDist", { "p_pos", "p_vel", "p_predPos" });

  clContext.releaseGLBuffers({ "p_pos", "c_partDetector", "u_cameraPos" });
}

/*

    //std::vector<std::array<float, 4>> predPos(m_maxNbParticles, { 0, 0, 0, 0 });
    //clContext.unloadBufferFromDevice("p_predPos", 0, sizeof(float) * predPos.size(), predPos.data());

    //std::vector<std::array<float, 4>> vel(m_maxNbParticles, { 0, 0, 0, 0 });
    //clContext.unloadBufferFromDevice("p_vel", 0, sizeof(float) * vel.size(), vel.data());

void Fluids::print(const std::string& name, int nbItems)
{
  CL::Context& clContext = CL::Context::Get();
  std::vector<std::array<float, 4>> pos(m_maxNbParticles, { 0, 0, 0, 0 });
  clContext.unloadBufferFromDevice(name, 0, sizeof(float) * pos.size(), pos.data());
  std::for_each(pos.begin(), pos.begin() + nbItems, [](const std::array<float, 4>& p)
      { return LOG_INFO("{}, {}, {}, {}", p[0], p[1], p[2], p[3]); });
}

// WIP
      if (0)
      {
        std::vector<float> density(m_maxNbParticles, 0);
        clContext.unloadBufferFromDevice("p_density", 0, sizeof(float) * density.size(), density.data());

        std::vector<unsigned int> cellIDs(m_maxNbParticles, 0);
        clContext.unloadBufferFromDevice("p_cellID", 0, sizeof(unsigned int) * cellIDs.size(), cellIDs.data());

        std::vector<std::array<float, 4>> pos(m_maxNbParticles, { 0, 0, 0, 0 });
        clContext.unloadBufferFromDevice("p_pos", 0, sizeof(float) * pos.size(), pos.data());
        std::vector<std::array<float, 4>> predPos(m_maxNbParticles, { 0, 0, 0, 0 });
        clContext.unloadBufferFromDevice("p_predPos", 0, sizeof(float) * predPos.size(), predPos.data());
        std::vector<std::array<float, 4>> corrPos(m_maxNbParticles, { 0, 0, 0, 0 });
        clContext.unloadBufferFromDevice("p_corrPos", 0, sizeof(float) * corrPos.size(), corrPos.data());

        std::vector<float> constFactor(m_maxNbParticles, 0);
        clContext.unloadBufferFromDevice("p_constFactor", 0, sizeof(float) * constFactor.size(), constFactor.data());

        std::vector<unsigned int> cameraDist(m_maxNbParticles, 0);
        clContext.unloadBufferFromDevice("p_cameraDist", 0, sizeof(unsigned int) * cameraDist.size(), cameraDist.data());

        int i = 0;
      }
    */