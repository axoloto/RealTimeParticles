#include "Boids.hpp"
#include "Geometry.hpp"
#include "Logging.hpp"
#include "Parameters.hpp"
#include "Utils.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace Physics;

#define PROGRAM_BOIDS "boids"

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

// boids.cl
#define KERNEL_RANDOM_POS "randPosVertsBoids"
#define KERNEL_FILL_COLOR "fillBoidsColor"
#define KERNEL_UPDATE_POS_BOUNCING "updatePosWithBouncingWalls"
#define KERNEL_UPDATE_POS_CYCLIC "updatePosWithCyclicWalls"
#define KERNEL_UPDATE_VEL "updateVel"
#define KERNEL_FILL_TEXT "fillBoidsTexture"
#define KERNEL_BOIDS_RULES_GRID_2D "applyBoidsRulesWithGrid2D"
#define KERNEL_BOIDS_RULES_GRID_3D "applyBoidsRulesWithGrid3D"
#define KERNEL_ADD_TARGET_RULE "addTargetRule"

Boids::Boids(ModelParams params)
    : Model(params)
    , m_scaleAlignment(1.6f)
    , m_scaleCohesion(1.45f)
    , m_scaleSeparation(1.6f)
    , m_activeAlignment(true)
    , m_activeSeparation(true)
    , m_activeCohesion(true)
    , m_simplifiedMode(true)
    , m_maxNbPartsInCell(3000)
    , m_radixSort(params.maxNbParticles)
    , m_target(std::make_unique<Target>(params.boxSize))
{
  m_currNbParticles = Utils::NbParticles::P512;

  createProgram();

  createBuffers();

  createKernels();

  m_init = true;

  reset();
}

bool Boids::createProgram() const
{
  CL::Context& clContext = CL::Context::Get();

  std::ostringstream clBuildOptions;
  clBuildOptions << "-DEFFECT_RADIUS_SQUARED=" << Utils::FloatToStr(1.0f * m_boxSize * m_boxSize / (m_gridRes * m_gridRes));
  clBuildOptions << " -DABS_WALL_POS=" << Utils::FloatToStr(m_boxSize / 2.0f);
  clBuildOptions << " -DGRID_RES=" << m_gridRes;
  clBuildOptions << " -DGRID_CELL_SIZE=" << Utils::FloatToStr(1.0f * m_boxSize / m_gridRes);
  clBuildOptions << " -DGRID_NUM_CELLS=" << m_nbCells;
  clBuildOptions << " -DNUM_MAX_PARTS_IN_CELL=" << m_maxNbPartsInCell;

  LOG_INFO(clBuildOptions.str());
  clContext.createProgram(PROGRAM_BOIDS, std::vector<std::string>({ "boids.cl", "utils.cl", "grid.cl" }), clBuildOptions.str());

  return true;
}

bool Boids::createBuffers() const
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createGLBuffer("u_cameraPos", m_cameraVBO, CL_MEM_READ_ONLY);
  clContext.createGLBuffer("p_pos", m_particlePosVBO, CL_MEM_READ_WRITE);
  clContext.createGLBuffer("p_col", m_particleColVBO, CL_MEM_READ_WRITE);
  clContext.createGLBuffer("c_partDetector", m_gridVBO, CL_MEM_READ_WRITE);

  clContext.createBuffer("p_vel", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_acc", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cellID", m_maxNbParticles * sizeof(unsigned int), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cameraDist", m_maxNbParticles * sizeof(unsigned int), CL_MEM_READ_WRITE);

  clContext.createBuffer("c_startEndPartID", 2 * m_nbCells * sizeof(unsigned int), CL_MEM_READ_WRITE);

  return true;
}

bool Boids::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();

  // Init only
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_INFINITE_POS, { "p_pos" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_RANDOM_POS, { "p_pos", "p_vel" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_COLOR, { "p_col" });

  // For rendering purpose only
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_RESET_PART_DETECTOR, { "c_partDetector" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_PART_DETECTOR, { "p_pos", "c_partDetector" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_RESET_CAMERA_DIST, { "p_cameraDist" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_CAMERA_DIST, { "p_pos", "u_cameraPos", "p_cameraDist" });

  // Boids Physics
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_VEL, { "p_acc", "", "", "p_vel" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_POS_BOUNCING, { "p_vel", "", "p_pos" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_UPDATE_POS_CYCLIC, { "p_vel", "", "p_pos" });

  // Radix Sort based on 3D grid
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_RESET_CELL_ID, { "p_cellID" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_CELL_ID, { "p_pos", "p_cellID" });

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_RESET_START_END_CELL, { "c_startEndPartID" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_START_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_FILL_END_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_ADJUST_END_CELL, { "c_startEndPartID" });

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_BOIDS_RULES_GRID_2D, { "p_pos", "p_vel", "c_startEndPartID", "", "p_acc" });
  clContext.createKernel(PROGRAM_BOIDS, KERNEL_BOIDS_RULES_GRID_3D, { "p_pos", "p_vel", "c_startEndPartID", "", "p_acc" });

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_ADD_TARGET_RULE, { "p_pos", "", "", "p_acc", "p_acc" });

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
  boidsParams[4] = isTargetActivated() ? 1.0f : 0.0f;
  clContext.setKernelArg(KERNEL_BOIDS_RULES_GRID_2D, 3, sizeof(boidsParams), &boidsParams);
  clContext.setKernelArg(KERNEL_BOIDS_RULES_GRID_3D, 3, sizeof(boidsParams), &boidsParams);

  if (isTargetActivated())
  {
    const auto squaredRadiusEffect = targetRadiusEffect() * targetRadiusEffect();
    const auto signEffect = targetSignEffect();
    clContext.setKernelArg(KERNEL_ADD_TARGET_RULE, 2, sizeof(float), &squaredRadiusEffect);
    clContext.setKernelArg(KERNEL_ADD_TARGET_RULE, 3, sizeof(int), &signEffect);
  }
}

void Boids::reset()
{
  if (!m_init)
    return;

  updateBoidsParamsInKernel();

  CL::Context& clContext = CL::Context::Get();

  initBoidsParticles();

  clContext.acquireGLBuffers({ "p_pos", "p_col", "c_partDetector" });

  clContext.runKernel(KERNEL_FILL_COLOR, m_maxNbParticles);
  clContext.runKernel(KERNEL_RESET_PART_DETECTOR, m_nbCells);
  clContext.runKernel(KERNEL_FILL_PART_DETECTOR, m_currNbParticles);

  clContext.runKernel(KERNEL_RESET_CELL_ID, m_maxNbParticles);
  clContext.runKernel(KERNEL_RESET_CAMERA_DIST, m_maxNbParticles);

  clContext.releaseGLBuffers({ "p_pos", "p_col", "c_partDetector" });
}

void Boids::initBoidsParticles()
{
  if (m_currNbParticles > m_maxNbParticles)
  {
    LOG_ERROR("Cannot init boids, current number of particles is higher than max limit");
    return;
  }

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_pos" });

  std::vector<Math::float3> gridVerts;

  if (m_dimension == Dimension::dim2D)
  {
    const auto& subdiv2D = Utils::GetNbParticlesSubdiv2D((Utils::NbParticles)m_currNbParticles);
    Math::int2 grid2DRes = { subdiv2D[0], subdiv2D[1] };
    Math::float3 start2D = { 0.0f, m_boxSize / -6.0f, m_boxSize / -6.0f };
    Math::float3 end2D = { 0.0f, m_boxSize / 6.0f, m_boxSize / 6.0f };

    gridVerts = Geometry::Generate2DGrid(Geometry::Shape2D::Circle, Geometry::Plane::YZ, grid2DRes, start2D, end2D);
  }
  else if (m_dimension == Dimension::dim3D)
  {
    const auto& subdiv3D = Utils::GetNbParticlesSubdiv3D((Utils::NbParticles)m_currNbParticles);
    Math::int3 grid3DRes = { subdiv3D[0], subdiv3D[1], subdiv3D[2] };
    Math::float3 start3D = { m_boxSize / -6.0f, m_boxSize / -6.0f, m_boxSize / -6.0f };
    Math::float3 end3D = { m_boxSize / 6.0f, m_boxSize / 6.0f, m_boxSize / 6.0f };

    gridVerts = Geometry::Generate3DGrid(Geometry::Shape3D::Sphere, grid3DRes, start3D, end3D);
  }

  const float& inf = std::numeric_limits<float>::infinity();
  std::vector<std::array<float, 4>> pos(m_maxNbParticles, std::array<float, 4>({ inf, inf, inf, 0.0f }));

  std::transform(gridVerts.cbegin(), gridVerts.cend(), pos.begin(),
      [](const Math::float3& vertPos) -> std::array<float, 4> { return { vertPos.x, vertPos.y, vertPos.z, 0.0f }; });

  clContext.loadBufferFromHost("p_pos", 0, 4 * sizeof(float) * pos.size(), pos.data());
  // Using same buffer to initialize vel, giving interesting patterns
  clContext.loadBufferFromHost("p_vel", 0, 4 * sizeof(float) * pos.size(), pos.data());

  clContext.releaseGLBuffers({ "p_pos" });
}

void Boids::update()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_pos", "p_col", "c_partDetector", "u_cameraPos" });

  if (!m_pause)
  {
    float timeStep = 0.1f;
    clContext.runKernel(KERNEL_FILL_CELL_ID, m_currNbParticles);

    m_radixSort.sort("p_cellID", { "p_pos", "p_col", "p_vel", "p_acc" });

    clContext.runKernel(KERNEL_RESET_START_END_CELL, m_nbCells);
    clContext.runKernel(KERNEL_FILL_START_CELL, m_currNbParticles);
    clContext.runKernel(KERNEL_FILL_END_CELL, m_currNbParticles);

    if (m_simplifiedMode)
      clContext.runKernel(KERNEL_ADJUST_END_CELL, m_nbCells);

    if (m_dimension == Dimension::dim2D)
      clContext.runKernel(KERNEL_BOIDS_RULES_GRID_2D, m_currNbParticles);
    else
      clContext.runKernel(KERNEL_BOIDS_RULES_GRID_3D, m_currNbParticles);

    if (isTargetActivated())
    {
      m_target->updatePos(m_dimension, m_velocity);
      auto targetXYZ = m_target->pos();
      std::array<float, 4> targetPos = { targetXYZ.x, targetXYZ.y, targetXYZ.z, 0.0f };
      clContext.setKernelArg(KERNEL_ADD_TARGET_RULE, 1, sizeof(float) * 4, &targetPos);
      clContext.runKernel(KERNEL_ADD_TARGET_RULE, m_currNbParticles);
    }

    clContext.setKernelArg(KERNEL_UPDATE_VEL, 1, sizeof(float), &timeStep);
    clContext.runKernel(KERNEL_UPDATE_VEL, m_currNbParticles);

    switch (m_boundary)
    {
    case Boundary::CyclicWall:
      clContext.setKernelArg(KERNEL_UPDATE_POS_CYCLIC, 1, sizeof(float), &timeStep);
      clContext.runKernel(KERNEL_UPDATE_POS_CYCLIC, m_currNbParticles);
      break;
    case Boundary::BouncingWall:
      clContext.setKernelArg(KERNEL_UPDATE_POS_BOUNCING, 1, sizeof(float), &timeStep);
      clContext.runKernel(KERNEL_UPDATE_POS_BOUNCING, m_currNbParticles);
      break;
    }

    clContext.runKernel(KERNEL_RESET_PART_DETECTOR, m_nbCells);
    clContext.runKernel(KERNEL_FILL_PART_DETECTOR, m_currNbParticles);
  }

  clContext.runKernel(KERNEL_FILL_CAMERA_DIST, m_currNbParticles);

  m_radixSort.sort("p_cameraDist", { "p_pos", "p_col", "p_vel", "p_acc" });

  clContext.releaseGLBuffers({ "p_pos", "p_col", "c_partDetector", "u_cameraPos" });
}