#include "Boids.hpp"
#include "Geometry.hpp"
#include "Logging.hpp"
#include "Parameters.hpp"
#include "Utils.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace Physics::CL;

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
#define KERNEL_FILL_COLOR "bd_fillBoidsColor"
#define KERNEL_UPDATE_POS_BOUNCING "bd_updatePosAndApplyWallBC"
#define KERNEL_UPDATE_POS_CYCLIC "bd_updatePosAndApplyPeriodicBC"
#define KERNEL_UPDATE_VEL "bd_updateVel"
#define KERNEL_FILL_TEXT "fillBoidsTexture"
#define KERNEL_BOIDS_RULES_GRID_2D "bd_applyBoidsRulesWithGrid2D"
#define KERNEL_BOIDS_RULES_GRID_3D "bd_applyBoidsRulesWithGrid3D"
#define KERNEL_ADD_TARGET_RULE "bd_addTargetRule"

static const json initBoidsJson // clang-format off
{ 
  {"Boids", {
      { "Velocity", { 0.5f, 0.01f, 5.0f } },
      { "Target",
        {
          { "Enable##Target", false },
          { "Show", true },
          { "Radius", { 2.0f, 1.0f, 20.0f } },
          { "Attract", true}
        }
      },
      { "Alignment",
        {
          { "Enable##Alignment", true},
          { "Scale##Alignment", { 1.6f, 0.0f, 3.0f} },
        }
      },
      { "Cohesion", 
        {
          { "Enable##Cohesion", true},
          { "Scale##Cohesion", { 1.45f, 0.0f, 3.0f} },
        }
      },           
      { "Separation", 
        {
          { "Enable##Separation", true},
          { "Scale##Separation", { 1.6f, 0.0f, 3.0f} },
        }
      }
    }
  }
}; // clang-format on

Boids::Boids(ModelParams params)
    : OclModel<BoidsRuleKernelInputs, TargetKernelInputs>(params, BoidsRuleKernelInputs {}, TargetKernelInputs {}, json(initBoidsJson))
    , m_simplifiedMode(true)
    , m_maxNbPartsInCell(3000)
    , m_radixSort(params.maxNbParticles)
    , m_target(params.boxSize.x)
{
  createProgram();

  createBuffers();

  createKernels();

  m_init = true;

  reset();
}

// Must be defined on implementation side to have RadixSort complete
Boids::~Boids() {};

bool Boids::createProgram() const
{
  CL::Context& clContext = CL::Context::Get();

  assert(m_boxSize.x / m_gridRes.x == m_boxSize.y / m_gridRes.y);
  assert(m_boxSize.z / m_gridRes.z == m_boxSize.y / m_gridRes.y);

  std::ostringstream clBuildOptions;
  clBuildOptions << "-DEFFECT_RADIUS_SQUARED=" << Utils::FloatToStr(1.0f * m_boxSize.x * m_boxSize.x / (m_gridRes.x * m_gridRes.x));
  clBuildOptions << " -DABS_WALL_X=" << Utils::FloatToStr(m_boxSize.x / 2.0f);
  clBuildOptions << " -DABS_WALL_Y=" << Utils::FloatToStr(m_boxSize.y / 2.0f);
  clBuildOptions << " -DABS_WALL_Z=" << Utils::FloatToStr(m_boxSize.z / 2.0f);
  clBuildOptions << " -DGRID_RES_X=" << m_gridRes.x;
  clBuildOptions << " -DGRID_RES_Y=" << m_gridRes.y;
  clBuildOptions << " -DGRID_RES_Z=" << m_gridRes.z;
  clBuildOptions << " -DGRID_CELL_SIZE_XYZ=" << Utils::FloatToStr((float)m_boxSize.x / m_gridRes.x);
  clBuildOptions << " -DGRID_NUM_CELLS=" << m_nbCells;
  clBuildOptions << " -DNUM_MAX_PARTS_IN_CELL=" << m_maxNbPartsInCell;

  LOG_INFO(clBuildOptions.str());
  // file.cl order matters, define.cl must be first
  clContext.createProgram(PROGRAM_BOIDS, std::vector<std::string>({ "define.cl", "boids.cl", "utils.cl", "grid.cl" }), clBuildOptions.str());

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

  clContext.createKernel(PROGRAM_BOIDS, KERNEL_ADD_TARGET_RULE, { "p_pos", "", "", "p_acc" });

  return true;
}

void Boids::transferJsonInputsToModel(json& inputJson)
{
  if (!m_init)
    return;

  try
  {
    const auto& boidsJson = inputJson["Boids"];

    auto& boidsRuleKernelInputs = getKernelInput<BoidsRuleKernelInputs>(0);

    boidsRuleKernelInputs.velocityScale = (cl_float)(boidsJson["Velocity"][0]);
    boidsRuleKernelInputs.alignmentScale = boidsJson["Alignment"]["Enable##Alignment"] ? (cl_float)(boidsJson["Alignment"]["Scale##Alignment"][0]) : 0.0f;
    boidsRuleKernelInputs.separationScale = boidsJson["Separation"]["Enable##Separation"] ? (cl_float)(boidsJson["Separation"]["Scale##Separation"][0]) : 0.0f;
    boidsRuleKernelInputs.cohesionScale = boidsJson["Cohesion"]["Enable##Cohesion"] ? (cl_float)(boidsJson["Cohesion"]["Scale##Cohesion"][0]) : 0.0f;

    auto& targetKernelInputs = getKernelInput<TargetKernelInputs>(1);

    m_target.activate(boidsJson["Target"]["Enable##Target"]);
    m_target.show(boidsJson["Target"]["Show"]);

    m_target.setRadiusEffect(boidsJson["Target"]["Radius"][0]);
    targetKernelInputs.targetRadiusEffect = m_target.radiusEffect();

    m_target.setSignEffect((int)(boidsJson["Target"]["Attract"]));
    targetKernelInputs.targetSignEffect = m_target.signEffect();
  }
  catch (...)
  {
    LOG_ERROR("Boids Input Json parsing is incorrect, did you use a wrong path for a parameter?");

    throw std::runtime_error("Wrong Json parsing");
  }
}

void Boids::transferKernelInputsToGPU()
{
  CL::Context& clContext = CL::Context::Get();

  const auto& boidsRuleKernelInputs = getKernelInput<BoidsRuleKernelInputs>(0);
  clContext.setKernelArg(KERNEL_UPDATE_VEL, 2, sizeof(float), &boidsRuleKernelInputs.velocityScale);
  clContext.setKernelArg(KERNEL_BOIDS_RULES_GRID_2D, 3, sizeof(BoidsRuleKernelInputs), &boidsRuleKernelInputs);
  clContext.setKernelArg(KERNEL_BOIDS_RULES_GRID_3D, 3, sizeof(BoidsRuleKernelInputs), &boidsRuleKernelInputs);

  if (isTargetActivated())
  {
    const auto& targetKernelInputs = getKernelInput<TargetKernelInputs>(1);
    clContext.setKernelArg(KERNEL_ADD_TARGET_RULE, 2, sizeof(TargetKernelInputs), &targetKernelInputs);
  }
}

void Boids::reset()
{
  if (!m_init)
    return;

  resetInputJson(initBoidsJson);

  switch (m_case)
  {
  case Utils::PhysicsCase::BOIDS_SMALL:
  {
    m_currNbParticles = Utils::NbParticles::P512;
    break;
  }
  case Utils::PhysicsCase::BOIDS_MEDIUM:
  {
    m_currNbParticles = Utils::NbParticles::P16K;
    break;
  }
  case Utils::PhysicsCase::BOIDS_LARGE:
  {
    m_currNbParticles = Utils::NbParticles::P65K;
    break;
  }
  case Utils::PhysicsCase::BOIDS_XLARGE:
  {
    m_currNbParticles = Utils::NbParticles::P130K;
    break;
  }
  }

  updateModelWithInputJson(getInputJson());

  initBoidsParticles();

  CL::Context& clContext = CL::Context::Get();

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

  if (m_dimension == Geometry::Dimension::dim2D)
  {
    const auto& subdiv2D = Utils::GetNbParticlesSubdiv2D((Utils::NbParticles)m_currNbParticles);
    Math::int2 grid2DRes = { subdiv2D[0], subdiv2D[1] };
    Math::float3 start2D = { 0.0f, m_boxSize.y / -6.0f, m_boxSize.z / -6.0f };
    Math::float3 end2D = { 0.0f, m_boxSize.y / 6.0f, m_boxSize.z / 6.0f };

    gridVerts = Geometry::Generate2DGrid(Geometry::Shape2D::Circle, Geometry::Plane::YZ, grid2DRes, start2D, end2D);
  }
  else if (m_dimension == Geometry::Dimension::dim3D)
  {
    const auto& subdiv3D = Utils::GetNbParticlesSubdiv3D((Utils::NbParticles)m_currNbParticles);
    Math::int3 grid3DRes = { subdiv3D[0], subdiv3D[1], subdiv3D[2] };
    Math::float3 start3D = { m_boxSize.x / -6.0f, m_boxSize.y / -6.0f, m_boxSize.z / -6.0f };
    Math::float3 end3D = { m_boxSize.x / 6.0f, m_boxSize.y / 6.0f, m_boxSize.z / 6.0f };

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

    if (m_dimension == Geometry::Dimension::dim2D)
      clContext.runKernel(KERNEL_BOIDS_RULES_GRID_2D, m_currNbParticles);
    else
      clContext.runKernel(KERNEL_BOIDS_RULES_GRID_3D, m_currNbParticles);

    if (isTargetActivated())
    {
      m_target.updatePos(m_dimension, getKernelInput<BoidsRuleKernelInputs>(0).velocityScale);
      auto targetXYZ = m_target.pos();
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