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
#define KERNEL_RESET_PART_DETECTOR "resetGridDetector"
#define KERNEL_FILL_PART_DETECTOR "fillGridDetector"
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
#define KERNEL_FILL_COLOR "fillFluidColor"

Fluids::Fluids(ModelParams params)
    : Model(params)
    , m_simplifiedMode(true)
    , m_maxNbPartsInCell(1000)
    , m_radixSort(params.maxNbParticles)
    , m_initialCase(CaseType::DAM)
    , m_nbJacobiIters(3)
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
  clBuildOptions << " -DSPIKY_COEFF=" << Utils::FloatToStr(15.0f / (Math::PI_F * std::powf(effectRadius, 6)));

  LOG_INFO(clBuildOptions.str());
  clContext.createProgram(PROGRAM_FLUIDS, std::vector<std::string>({ "fluids.cl", "utils.cl", "grid.cl" }), clBuildOptions.str());

  return true;
}

bool Fluids::createBuffers() const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createGLBuffer("u_cameraPos", m_cameraVBO, CL_MEM_READ_ONLY);
  clContext.createGLBuffer("p_pos", m_particlePosVBO, CL_MEM_READ_WRITE);
  clContext.createGLBuffer("p_col", m_particleColVBO, CL_MEM_READ_WRITE);
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
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RANDOM_POS, { "", "p_pos", "p_vel" });

  // For rendering purpose only
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RESET_PART_DETECTOR, { "c_partDetector" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_PART_DETECTOR, { "p_pos", "c_partDetector" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RESET_CAMERA_DIST, { "p_cameraDist" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_CAMERA_DIST, { "p_pos", "u_cameraPos", "p_cameraDist" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_COLOR, { "p_density", "", "p_col" });

  // Radix Sort based on 3D grid, using predicted positions, not corrected ones
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RESET_CELL_ID, { "p_cellID" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_CELL_ID, { "p_predPos", "p_cellID" });

  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_RESET_START_END_CELL, { "c_startEndPartID" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_START_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_FILL_END_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_ADJUST_END_CELL, { "c_startEndPartID" });

  // Position Based Fluids
  /// Position prediction
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_PREDICT_POS, { "p_pos", "p_vel", "", "p_predPos" });
  /// Boundary conditions
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_APPLY_BOUNDARY, { "p_predPos" });
  /// Jacobi solver to correct position
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_DENSITY, { "p_predPos", "c_startEndPartID", "", "p_density" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_CONSTRAINT_FACTOR, { "p_predPos", "p_density", "c_startEndPartID", "", "p_constFactor" });
  clContext.createKernel(PROGRAM_FLUIDS, KERNEL_CONSTRAINT_CORRECTION, { "p_constFactor", "c_startEndPartID", "p_predPos", "", "p_corrPos" });
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

  //m_kernelInputs.poly6Coeff = 315.0f / (64.0f * Math::PI_F * std::powf(effectRadius, 9));
  //m_kernelInputs.spikyCoeff = 15.0f / (Math::PI_F * std::powf(effectRadius, 6));

  m_kernelInputs.dim = (m_dimension == Dimension::dim2D) ? 2 : 3;

  const float effectRadius = ((float)m_boxSize) / m_gridRes;
  m_kernelInputs.effectRadius = effectRadius;

  clContext.setKernelArg(KERNEL_RANDOM_POS, 0, sizeof(FluidKernelInputs), &m_kernelInputs);
  clContext.setKernelArg(KERNEL_PREDICT_POS, 2, sizeof(FluidKernelInputs), &m_kernelInputs);
  clContext.setKernelArg(KERNEL_UPDATE_VEL, 2, sizeof(FluidKernelInputs), &m_kernelInputs);
  clContext.setKernelArg(KERNEL_DENSITY, 2, sizeof(FluidKernelInputs), &m_kernelInputs);
  clContext.setKernelArg(KERNEL_CONSTRAINT_FACTOR, 3, sizeof(FluidKernelInputs), &m_kernelInputs);
  clContext.setKernelArg(KERNEL_CONSTRAINT_CORRECTION, 3, sizeof(FluidKernelInputs), &m_kernelInputs);
  clContext.setKernelArg(KERNEL_FILL_COLOR, 1, sizeof(FluidKernelInputs), &m_kernelInputs);
}

void Fluids::reset()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  updateFluidsParamsInKernel();

  initFluid();

  clContext.acquireGLBuffers({ "p_pos", "c_partDetector" });
  clContext.runKernel(KERNEL_RESET_PART_DETECTOR, m_nbCells);
  clContext.runKernel(KERNEL_FILL_PART_DETECTOR, m_currNbParticles);
  clContext.releaseGLBuffers({ "p_pos", "c_partDetector" });

  clContext.runKernel(KERNEL_RESET_CELL_ID, m_maxNbParticles);
  clContext.runKernel(KERNEL_RESET_CAMERA_DIST, m_maxNbParticles);
}

void Fluids::initFluid() const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_pos", "p_col" });

  //float dim = (m_dimension == Dimension::dim2D) ? 2.0f : 3.0f;

  // Dam setup
  float inf = std::numeric_limits<float>::infinity();
  std::vector<std::array<float, 4>> pos(m_maxNbParticles, std::array<float, 4>({ inf, inf, inf, 0.0f }));

  int i = 0;
  float effectRadius = ((float)m_boxSize) / m_gridRes;
  float gridSpacing = 0.4f * effectRadius;
  Math::float3 startFluidPos = { 5.0f, 5.0f, 0.0f };
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

  std::vector<std::array<float, 4>> col(m_maxNbParticles, std::array<float, 4>({ 1.0f, 1.0f, 0.0f, 0.0f }));
  clContext.loadBufferFromHost("p_col", 0, 4 * sizeof(float) * col.size(), col.data());

  clContext.releaseGLBuffers({ "p_pos", "p_col" });
}

void Fluids::update()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_pos", "p_col", "c_partDetector", "u_cameraPos" });

  if (!m_pause)
  {
    // Predicting velocity and position
    clContext.runKernel(KERNEL_PREDICT_POS, m_currNbParticles);

    // NNS - spatial partitioning
    clContext.runKernel(KERNEL_FILL_CELL_ID, m_currNbParticles);

    m_radixSort.sort("p_cellID", { "p_pos", "p_col", "p_vel", "p_predPos" });

    clContext.runKernel(KERNEL_RESET_START_END_CELL, m_nbCells);
    clContext.runKernel(KERNEL_FILL_START_CELL, m_currNbParticles);
    clContext.runKernel(KERNEL_FILL_END_CELL, m_currNbParticles);

    if (m_simplifiedMode)
      clContext.runKernel(KERNEL_ADJUST_END_CELL, m_nbCells);

    // Correcting positions to fit constraints
    for (int iter = 0; iter < m_nbJacobiIters; ++iter)
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
    clContext.runKernel(KERNEL_RESET_PART_DETECTOR, m_nbCells);
    clContext.runKernel(KERNEL_FILL_PART_DETECTOR, m_currNbParticles);
    clContext.runKernel(KERNEL_FILL_COLOR, m_currNbParticles);
  }

  // Rendering purpose
  clContext.runKernel(KERNEL_FILL_CAMERA_DIST, m_currNbParticles);

  m_radixSort.sort("p_cameraDist", { "p_pos", "p_col", "p_vel", "p_predPos" });

  clContext.releaseGLBuffers({ "p_pos", "p_col", "c_partDetector", "u_cameraPos" });
}