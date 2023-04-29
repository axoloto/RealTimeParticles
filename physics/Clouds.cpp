#include "Clouds.hpp"
#include "Geometry.hpp"
#include "Logging.hpp"
#include "Parameters.hpp"
#include "Utils.hpp"

#include "ocl/Context.hpp"

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

#define PROGRAM_CLOUDS "Clouds"

// utils.cl
#define KERNEL_INFINITE_POS "infPosVerts"
#define KERNEL_RESET_CAMERA_DIST "resetCameraDist"
#define KERNEL_FILL_CAMERA_DIST "fillCameraDist"
#define KERNEL_FILL_COLOR "fillColorFloat"

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
#define KERNEL_APPLY_BOUNDARY "cld_applyBoundaryCondWithMixedWalls"
// #define KERNEL_APPLY_BOUNDARY "fld_applyBoundaryCondition"
#define KERNEL_DENSITY "fld_computeDensity"
#define KERNEL_CONSTRAINT_FACTOR_FLUIDS "fld_computeConstraintFactor"
#define KERNEL_CONSTRAINT_CORRECTION_FLUIDS "fld_computeConstraintCorrection"
#define KERNEL_CORRECT_POS "fld_correctPosition"
#define KERNEL_UPDATE_VEL "fld_updateVel"
#define KERNEL_COMPUTE_VORTICITY "fld_computeVorticity"
#define KERNEL_VORTICITY_CONFINEMENT "fld_applyVorticityConfinement"
#define KERNEL_XSPH_VISCOSITY "fld_applyXsphViscosityCorrection"
#define KERNEL_UPDATE_POS "fld_updatePosition"

// clouds.cl
#define KERNEL_INIT_TEMP "cld_initTemperature"
#define KERNEL_INIT_VAPOR_DENSITY "cld_initVaporDensity"
#define KERNEL_RANDOM_POS "cld_randPosVertsClouds"
#define KERNEL_HEAT_GROUND "cld_heatFromGround"
#define KERNEL_BUOYANCY "cld_computeBuoyancy"
#define KERNEL_ADIABATIC_COOLING "cld_applyAdiabaticCooling"
#define KERNEL_CLOUD_GENERATION "cld_generateCloud"
#define KERNEL_PHASE_TRANSITION "cld_applyPhaseTransition"
#define KERNEL_LATENT_HEAT "cld_applyLatentHeat"
#define KERNEL_PREDICT_POS "cld_predictPosition"
//
#define KERNEL_LAPLACIAN_TEMP "cld_computeLaplacianTemp"
#define KERNEL_CONSTRAINT_FACTOR_TEMP "cld_computeConstraintFactor"
#define KERNEL_CONSTRAINT_CORRECTION_TEMP "cld_computeConstraintCorrection"
#define KERNEL_CORRECT_TEMP "cld_correctTemperature"

namespace Physics
{
// Fluids params for Position Based Fluids part of clouds sim
struct FluidKernelInputs
{
  cl_float restDensity = 700.0f;
  cl_float relaxCFM = 600.0f;
  cl_float timeStep = 0.01f;
  cl_uint dim = 3;
  // Artifical pressure if enabled will try to reduce tensile instability
  cl_uint isArtPressureEnabled = 0;
  cl_float artPressureRadius = 0.006f;
  cl_float artPressureCoeff = 0.001f;
  cl_uint artPressureExp = 4;
  // Vorticity confinement if enabled will try to replace lost energy due to virtual damping
  cl_uint isVorticityConfEnabled = 0;
  cl_float vorticityConfCoeff = 0.0004f;
  cl_float xsphViscosityCoeff = 0.0001f;
};

// Clouds params for clouds-specific physics
struct CloudKernelInputs
{
  // Must be always equal to timeStep of FluidKernelInputs
  cl_float timeStep = 0.01f;
  // groundHeatCoeff * timeStep = max temperature increase due to heat source in ground for closest particles from ground
  cl_float groundHeatCoeff = 200.0f; // i.e here 2K/iteration
  cl_float buoyancyCoeff = 0.075f;
  cl_float adiabaticLapseRate = 80.0f;
  cl_float phaseTransitionRate = 100.0f;
  cl_float latentHeatCoeff = 0.003f;
  // Enable constraint on temperature field, forcing its Laplacian field to be null
  cl_uint isTempSmoothingEnabled = 1;
  cl_float restDensity = 700.0f; // must be equal to fluids one
  cl_float relaxCFM = 600.0f;
};

const std::map<Clouds::CaseType, std::string, Clouds::CompareCaseType> Clouds::ALL_CASES {
  { CaseType::CUMULUS, "Cumulus" },
  { CaseType::HOMOGENEOUS, "Homogeneous" }
};
}

Clouds::Clouds(ModelParams params)
    : Model(params)
    , m_simplifiedMode(true)
    , m_maxNbPartsInCell(100)
    , m_radixSort(params.maxNbParticles)
    , m_fluidKernelInputs(std::make_unique<FluidKernelInputs>())
    , m_cloudKernelInputs(std::make_unique<CloudKernelInputs>())
    , m_initialCase(CaseType::CUMULUS)
    , m_nbJacobiIters(2)
{
  createProgram();

  createBuffers();

  createKernels();

  m_init = (m_fluidKernelInputs && m_cloudKernelInputs && !m_allDisplayableQuantities.empty());

  reset();
}

// Must be on implementation side as FluidKernelInputs must be complete
Clouds::~Clouds() {};

bool Clouds::createProgram() const
{
  CL::Context& clContext = CL::Context::Get();

  assert(m_boxSize.x / m_gridRes.x == m_boxSize.y / m_gridRes.y);
  assert(m_boxSize.z / m_gridRes.z == m_boxSize.y / m_gridRes.y);

  float effectRadius = ((float)m_boxSize.x) / m_gridRes.x;

  std::ostringstream clBuildOptions;
  clBuildOptions << "-DEFFECT_RADIUS=" << Utils::FloatToStr(effectRadius);
  clBuildOptions << " -DABS_WALL_X=" << Utils::FloatToStr(m_boxSize.x / 2.0f);
  clBuildOptions << " -DABS_WALL_Y=" << Utils::FloatToStr(m_boxSize.y / 2.0f);
  clBuildOptions << " -DABS_WALL_Z=" << Utils::FloatToStr(m_boxSize.z / 2.0f);
  clBuildOptions << " -DGRID_RES_X=" << m_gridRes.x;
  clBuildOptions << " -DGRID_RES_Y=" << m_gridRes.y;
  clBuildOptions << " -DGRID_RES_Z=" << m_gridRes.z;
  clBuildOptions << " -DGRID_CELL_SIZE_XYZ=" << Utils::FloatToStr((float)m_boxSize.x / m_gridRes.x);
  clBuildOptions << " -DGRID_NUM_CELLS=" << m_nbCells;
  clBuildOptions << " -DNUM_MAX_PARTS_IN_CELL=" << m_maxNbPartsInCell;
  clBuildOptions << " -DPOLY6_COEFF=" << Utils::FloatToStr(315.0f / (64.0f * Math::PI_F * std::pow(effectRadius, 9.f)));
  clBuildOptions << " -DSPIKY_COEFF=" << Utils::FloatToStr(15.0f / (Math::PI_F * std::pow(effectRadius, 6.f)));
  clBuildOptions << " -DMAX_VEL=" << Utils::FloatToStr(30.0f);

  LOG_INFO(clBuildOptions.str());
  // file.cl order matters
  // 1/ define.cl must be first as it defines variables used by other kernels
  // 2/ fluids.cl contains Position Based Fluids algorithms needed for the fluids part of the cloud sim
  // 3/ clouds.cl contains Clouds-specific physics and constraint on temperature field, it needs PBF framework
  clContext.createProgram(PROGRAM_CLOUDS, std::vector<std::string>({ "define.cl", "fluids.cl", "clouds.cl", "grid.cl", "utils.cl" }), clBuildOptions.str());

  return true;
}

bool Clouds::createBuffers()
{
  CL::Context& clContext = CL::Context::Get();

  clContext.createGLBuffer("u_cameraPos", m_cameraVBO, CL_MEM_READ_ONLY);
  clContext.createGLBuffer("p_pos", m_particlePosVBO, CL_MEM_READ_WRITE);
  clContext.createGLBuffer("p_col", m_particleColVBO, CL_MEM_READ_WRITE);
  clContext.createGLBuffer("c_partDetector", m_gridVBO, CL_MEM_READ_WRITE);

  clContext.createBuffer("p_partID", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);

  // Position Based Fluids
  clContext.createBuffer("p_density", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_predPos", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_corrPos", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_constFactorFld", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_vel", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_velInViscosity", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_vort", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cellID", m_maxNbParticles * sizeof(unsigned int), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cameraDist", m_maxNbParticles * sizeof(unsigned int), CL_MEM_READ_WRITE);

  // Clouds specific
  // Some buffers are duplicated because they are both input/output of some kernels
  clContext.createBuffer("p_temp", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_tempIn", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_laplacianTemp", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_corrTemp", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_constFactorTemp", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_vaporDens", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_vaporDensIn", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cloudDens", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cloudDensIn", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_buoyancy", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cloudGen", m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);

  clContext.createBuffer("c_startEndPartID", 2 * m_nbCells * sizeof(unsigned int), CL_MEM_READ_WRITE);

  // Physical parameters displayable in UI, only m_maxNbParts * sizeof(float) size supported for now
  PhysicalQuantity partID { "Particle ID", "p_partID", { 0.0f, (float)(m_maxNbParticles - 1) }, { 0.0f, (float)(m_maxNbParticles - 1) } };
  m_allDisplayableQuantities.insert(std::make_pair(partID.name, partID));
  PhysicalQuantity vaporDens { "Vapor Density", "p_vaporDens", { 0.0f, 100.0f }, { 0.0f, 40.0f } };
  m_allDisplayableQuantities.insert(std::make_pair(vaporDens.name, vaporDens));
  PhysicalQuantity cloudDens { "Cloud Density", "p_cloudDens", { 0.0f, 100.0f }, { 0.0f, 50.0f } };
  m_allDisplayableQuantities.insert(std::make_pair(cloudDens.name, cloudDens));
  PhysicalQuantity netForce { "Net Force", "p_buoyancy", { -10.0f, 10.0f }, { -1.0f, 1.0f } };
  m_allDisplayableQuantities.insert(std::make_pair(netForce.name, netForce));
  PhysicalQuantity temp { "Temperature", "p_temp", { 0.0f, 2000.0f }, { 0.0f, 2000.0f } };
  m_allDisplayableQuantities.insert(std::make_pair(temp.name, temp));

  m_currentDisplayedQuantityName = partID.name;

  return true;
}

bool Clouds::createKernels() const
{
  CL::Context& clContext = CL::Context::Get();

  // Init only
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_INFINITE_POS, { "p_pos" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_RANDOM_POS, { "", "p_pos", "p_vel" });

  // For rendering purpose only
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_RESET_PART_DETECTOR, { "c_partDetector" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_PART_DETECTOR, { "p_pos", "c_partDetector" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_RESET_CAMERA_DIST, { "p_cameraDist" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_CAMERA_DIST, { "p_pos", "u_cameraPos", "p_cameraDist" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_COLOR, { "", "", "", "p_col" });

  // Radix Sort based on 3D grid, using predicted positions, not corrected ones
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_RESET_CELL_ID, { "p_cellID" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_CELL_ID, { "p_predPos", "p_cellID" });

  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_RESET_START_END_CELL, { "c_startEndPartID" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_START_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_END_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_ADJUST_END_CELL, { "c_startEndPartID" });

  // Clouds thermodynamics
  // Init steps
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_INIT_TEMP, { "p_pos", "p_temp" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_INIT_VAPOR_DENSITY, { "p_temp", "p_vaporDens" });
  //
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_HEAT_GROUND, { "p_tempIn", "p_pos", "", "p_temp" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_BUOYANCY, { "p_temp", "p_pos", "p_cloudDens", "", "p_buoyancy" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_ADIABATIC_COOLING, { "p_temp", "p_vel", "", "p_tempIn" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CLOUD_GENERATION, { "p_tempIn", "p_vaporDens", "p_cloudDens", "", "p_cloudGen" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_PHASE_TRANSITION, { "p_vaporDensIn", "p_cloudDensIn", "p_cloudGen", "", "p_vaporDens", "p_cloudDens" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_LATENT_HEAT, { "p_tempIn", "p_cloudGen", "", "p_temp" });
  // Jacobi solver to correct temperature
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_LAPLACIAN_TEMP, { "p_pos", "p_temp", "c_startEndPartID", "", "p_laplacianTemp" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CONSTRAINT_FACTOR_TEMP, { "p_pos", "p_laplacianTemp", "c_startEndPartID", "", "p_constFactorTemp" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CONSTRAINT_CORRECTION_TEMP, { "p_constFactorTemp", "c_startEndPartID", "p_pos", "", "p_corrTemp" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CORRECT_TEMP, { "p_corrTemp", "p_temp" });

  // Position Based Fluids - connected to clouds physics through buoyancy force applied on particles
  /// Boundary conditions
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_APPLY_BOUNDARY, { "p_pos" });
  /// Position prediction
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_PREDICT_POS, { "p_pos", "p_vel", "p_buoyancy", "", "p_predPos" });
  /// Jacobi solver to correct position
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_DENSITY, { "p_predPos", "c_startEndPartID", "", "p_density" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CONSTRAINT_FACTOR_FLUIDS, { "p_predPos", "p_density", "c_startEndPartID", "", "p_constFactorFld" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CONSTRAINT_CORRECTION_FLUIDS, { "p_constFactorFld", "c_startEndPartID", "p_predPos", "", "p_corrPos" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CORRECT_POS, { "p_corrPos", "p_predPos" });
  /// Velocity update and correction using vorticity confinement and xsph viscosity
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_UPDATE_VEL, { "p_predPos", "p_pos", "", "p_vel" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_COMPUTE_VORTICITY, { "p_predPos", "c_startEndPartID", "p_vel", "", "p_vort" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_VORTICITY_CONFINEMENT, { "p_predPos", "c_startEndPartID", "p_vort", "", "p_vel" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_XSPH_VISCOSITY, { "p_predPos", "c_startEndPartID", "p_velInViscosity", "", "p_vel" });
  /// Position update
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_UPDATE_POS, { "p_predPos", "p_pos" });

  return true;
}

void Clouds::updateFluidsParamsInKernels()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  m_fluidKernelInputs->dim = (m_dimension == Geometry::Dimension::dim2D) ? 2 : 3;

  clContext.setKernelArg(KERNEL_RANDOM_POS, 0, sizeof(FluidKernelInputs), m_fluidKernelInputs.get());
  clContext.setKernelArg(KERNEL_UPDATE_VEL, 2, sizeof(FluidKernelInputs), m_fluidKernelInputs.get());
  clContext.setKernelArg(KERNEL_DENSITY, 2, sizeof(FluidKernelInputs), m_fluidKernelInputs.get());
  clContext.setKernelArg(KERNEL_CONSTRAINT_FACTOR_FLUIDS, 3, sizeof(FluidKernelInputs), m_fluidKernelInputs.get());
  clContext.setKernelArg(KERNEL_CONSTRAINT_CORRECTION_FLUIDS, 3, sizeof(FluidKernelInputs), m_fluidKernelInputs.get());
  clContext.setKernelArg(KERNEL_COMPUTE_VORTICITY, 3, sizeof(FluidKernelInputs), m_fluidKernelInputs.get());
  clContext.setKernelArg(KERNEL_VORTICITY_CONFINEMENT, 3, sizeof(FluidKernelInputs), m_fluidKernelInputs.get());
  clContext.setKernelArg(KERNEL_XSPH_VISCOSITY, 3, sizeof(FluidKernelInputs), m_fluidKernelInputs.get());
}

void Clouds::updateCloudsParamsInKernels()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  clContext.setKernelArg(KERNEL_HEAT_GROUND, 2, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_BUOYANCY, 3, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_ADIABATIC_COOLING, 2, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_CLOUD_GENERATION, 3, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_PHASE_TRANSITION, 3, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_LATENT_HEAT, 2, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_PREDICT_POS, 3, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_LAPLACIAN_TEMP, 3, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_CONSTRAINT_FACTOR_TEMP, 3, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
  clContext.setKernelArg(KERNEL_CONSTRAINT_CORRECTION_TEMP, 3, sizeof(CloudKernelInputs), m_cloudKernelInputs.get());
}

void Clouds::reset()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  updateFluidsParamsInKernels();
  updateCloudsParamsInKernels();

  initCloudsParticles();

  clContext.acquireGLBuffers({ "p_pos", "c_partDetector" });
  clContext.runKernel(KERNEL_RESET_PART_DETECTOR, m_nbCells);
  clContext.runKernel(KERNEL_FILL_PART_DETECTOR, m_currNbParticles);
  clContext.releaseGLBuffers({ "p_pos", "c_partDetector" });

  clContext.runKernel(KERNEL_RESET_CELL_ID, m_maxNbParticles);
  clContext.runKernel(KERNEL_RESET_CAMERA_DIST, m_maxNbParticles);
}

void Clouds::initCloudsParticles()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_pos", "p_col" });

  std::vector<Math::float3> gridVerts;

  Math::float3 startFluidPos = { 0.0f, 0.0f, 0.0f };
  Math::float3 endFluidPos = { 0.0f, 0.0f, 0.0f };

  if (m_dimension == Geometry::Dimension::dim2D)
  {
    Geometry::Shape2D shape = Geometry::Shape2D::Rectangle;

    switch (m_initialCase)
    {
    case CaseType::CUMULUS:
      m_currNbParticles = Utils::NbParticles::P4K;
      shape = Geometry::Shape2D::Rectangle;
      startFluidPos = { 0.0f, m_boxSize.y / -2.0f, m_boxSize.z / -2.0f };
      endFluidPos = { 0.0f, 0.0f, m_boxSize.z / 2.0f };
      break;
    case CaseType::HOMOGENEOUS:
      m_currNbParticles = Utils::NbParticles::P4K;
      shape = Geometry::Shape2D::Rectangle;
      startFluidPos = { 0.0f, m_boxSize.y / -2.0f, m_boxSize.z / -2.0f };
      endFluidPos = { 0.0f, m_boxSize.y / 2.0f, m_boxSize.z / 2.0f };
      break;
    default:
      LOG_ERROR("Unkown case type");
      break;
    }

    const auto& subdiv2D = Utils::GetNbParticlesSubdiv2D((Utils::NbParticles)m_currNbParticles);
    Math::int2 grid2DRes = { subdiv2D[0], subdiv2D[1] };

    gridVerts = Geometry::Generate2DGrid(shape, Geometry::Plane::YZ, grid2DRes, startFluidPos, endFluidPos);
  }
  else if (m_dimension == Geometry::Dimension::dim3D)
  {
    Geometry::Shape3D shape = Geometry::Shape3D::Box;

    switch (m_initialCase)
    {
    case CaseType::CUMULUS:
      m_currNbParticles = Utils::NbParticles::P130K;
      shape = Geometry::Shape3D::Box;
      startFluidPos = { m_boxSize.x / -2.0f, m_boxSize.y / -2.0f, m_boxSize.z / -2.0f };
      endFluidPos = { m_boxSize.x / 2.0f, 0.0f, m_boxSize.z / 2.0f };
      break;
    case CaseType::HOMOGENEOUS:
      m_currNbParticles = Utils::NbParticles::P65K;
      shape = Geometry::Shape3D::Box;
      startFluidPos = { m_boxSize.x / -2.0f, m_boxSize.y / -2.0f, m_boxSize.z / -2.0f };
      endFluidPos = { m_boxSize.x / 2.0f, m_boxSize.y / 2.0f, m_boxSize.z / 2.0f };
      break;
    default:
      LOG_ERROR("Unkown case type");
      break;
    }

    const auto& subdiv3D = Utils::GetNbParticlesSubdiv3D((Utils::NbParticles)m_currNbParticles);
    Math::int3 grid3DRes = { subdiv3D[0], subdiv3D[1], subdiv3D[2] };

    gridVerts = Geometry::Generate3DGrid(shape, grid3DRes, startFluidPos, endFluidPos);
  }

  float inf = std::numeric_limits<float>::infinity();
  std::vector<std::array<float, 4>> pos(m_maxNbParticles, std::array<float, 4>({ inf, inf, inf, 0.0f }));

  std::transform(gridVerts.cbegin(), gridVerts.cend(), pos.begin(),
      [](const Math::float3& vertPos) -> std::array<float, 4> { return { vertPos.x, vertPos.y, vertPos.z, 0.0f }; });
  clContext.loadBufferFromHost("p_pos", 0, 4 * sizeof(float) * pos.size(), pos.data());

  std::vector<std::array<float, 4>> vel(m_maxNbParticles, std::array<float, 4>({ 0.0f, 0.0f, 0.0f, 0.0f }));
  clContext.loadBufferFromHost("p_vel", 0, 4 * sizeof(float) * vel.size(), vel.data());

  std::vector<std::array<float, 4>> col(m_maxNbParticles, std::array<float, 4>({ 0.0f, 0.1f, 1.0f, 0.0f }));
  clContext.loadBufferFromHost("p_col", 0, 4 * sizeof(float) * col.size(), col.data());

  std::vector<float> cloudDens(m_maxNbParticles, 0.0f);
  clContext.loadBufferFromHost("p_cloudDens", 0, sizeof(float) * cloudDens.size(), cloudDens.data());

  std::vector<float> partID(m_maxNbParticles, 0.0f);
  for (int i = 0; i != partID.size(); ++i)
    partID[i] = (float)i;
  clContext.loadBufferFromHost("p_partID", 0, sizeof(float) * partID.size(), partID.data());

  // Temperature field must be initialized before vapor density
  clContext.runKernel(KERNEL_INIT_TEMP, m_maxNbParticles);

  clContext.runKernel(KERNEL_INIT_VAPOR_DENSITY, m_maxNbParticles);

  clContext.releaseGLBuffers({ "p_pos", "p_col" });
}

void Clouds::update()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  clContext.acquireGLBuffers({ "p_pos", "p_col", "c_partDetector", "u_cameraPos" });

  if (!m_pause)
  {
    // Clouds thermodynamics
    // Copying temperature to other buffer as HeatGround kernel need it as both input and output
    clContext.copyBuffer("p_temp", "p_tempIn");
    clContext.runKernel(KERNEL_HEAT_GROUND, m_currNbParticles);
    // Computing buoyancy and gravity forces exerced on particles
    clContext.runKernel(KERNEL_BUOYANCY, m_currNbParticles);
    // Computing adiabatic cooling due to altitude increase
    clContext.runKernel(KERNEL_ADIABATIC_COOLING, m_currNbParticles);
    // Computing cloud generation value
    clContext.runKernel(KERNEL_CLOUD_GENERATION, m_currNbParticles);
    // Copying vapor and cloud density values to other buffers before running phase transition kernel using them as input/output
    clContext.copyBuffer("p_vaporDens", "p_vaporDensIn");
    clContext.copyBuffer("p_cloudDens", "p_cloudDensIn");
    clContext.runKernel(KERNEL_PHASE_TRANSITION, m_currNbParticles);
    //
    clContext.runKernel(KERNEL_LATENT_HEAT, m_currNbParticles);

    // Apply boundary only once per frame for now
    clContext.runKernel(KERNEL_APPLY_BOUNDARY, m_currNbParticles);

    // Predicting velocity and position
    // Step coupling fluids and clouds physics
    // where we apply clouds buoyancy and gravity forces on fluids particles
    clContext.runKernel(KERNEL_PREDICT_POS, m_currNbParticles);

    // NNS - spatial partitioning
    clContext.runKernel(KERNEL_FILL_CELL_ID, m_currNbParticles);

    m_radixSort.sort("p_cellID", { "p_pos", "p_col", "p_vel", "p_predPos" }, { "p_temp", "p_buoyancy", "p_vaporDens", "p_cloudDens", "p_partID" });

    clContext.runKernel(KERNEL_RESET_START_END_CELL, m_nbCells);
    clContext.runKernel(KERNEL_FILL_START_CELL, m_currNbParticles);
    clContext.runKernel(KERNEL_FILL_END_CELL, m_currNbParticles);

    if (m_simplifiedMode)
      clContext.runKernel(KERNEL_ADJUST_END_CELL, m_nbCells);

    // Apply constraint on temperature field in a similar way than position based fluids constraint on mass
    // This time, the constraint aims to homogenize temperature field, forcing its Laplacian field to be null
    if (m_cloudKernelInputs->isTempSmoothingEnabled)
    {
      for (int iter = 0; iter < 1; ++iter)
      {
        // Computing Laplacian of temperature field using SPH method, it is the constrained variable
        clContext.runKernel(KERNEL_LAPLACIAN_TEMP, m_currNbParticles);
        // Computing constraint factor Lambda
        clContext.runKernel(KERNEL_CONSTRAINT_FACTOR_TEMP, m_currNbParticles);
        // Computing constraint correction
        clContext.runKernel(KERNEL_CONSTRAINT_CORRECTION_TEMP, m_currNbParticles);
        // Applying correction on temperature field
        clContext.runKernel(KERNEL_CORRECT_TEMP, m_currNbParticles);
      }
    }

    // Correcting positions to fit constraints
    for (int iter = 0; iter < m_nbJacobiIters; ++iter)
    {
      // Clamping to boundary
      //clContext.runKernel(KERNEL_APPLY_BOUNDARY, m_currNbParticles);
      // Computing density using SPH method
      clContext.runKernel(KERNEL_DENSITY, m_currNbParticles);
      // Computing constraint factor Lambda
      clContext.runKernel(KERNEL_CONSTRAINT_FACTOR_FLUIDS, m_currNbParticles);
      // Computing position correction
      clContext.runKernel(KERNEL_CONSTRAINT_CORRECTION_FLUIDS, m_currNbParticles);
      // Correcting predicted position
      clContext.runKernel(KERNEL_CORRECT_POS, m_currNbParticles);
    }

    // Updating velocity
    clContext.runKernel(KERNEL_UPDATE_VEL, m_currNbParticles);

    if (m_fluidKernelInputs->isVorticityConfEnabled)
    {
      // Computing vorticity
      clContext.runKernel(KERNEL_COMPUTE_VORTICITY, m_currNbParticles);
      // Applying vorticity confinement to attenue virtual damping
      clContext.runKernel(KERNEL_VORTICITY_CONFINEMENT, m_currNbParticles);
      // Copying velocity buffer as input for vorticity confinement correction
      clContext.copyBuffer("p_vel", "p_velInViscosity");
      // Applying xsph viscosity correction for a more coherent motion
      clContext.runKernel(KERNEL_XSPH_VISCOSITY, m_currNbParticles);
    }

    // Updating pos
    clContext.runKernel(KERNEL_UPDATE_POS, m_currNbParticles);

    // Rendering purpose
    clContext.runKernel(KERNEL_RESET_PART_DETECTOR, m_nbCells);
    clContext.runKernel(KERNEL_FILL_PART_DETECTOR, m_currNbParticles);
  }

  // Sending selected physical parameter to color buffer for rendering
  const auto& currentPhysicalQuantity = currentDisplayedPhysicalQuantity();
  cl_float minVal = (cl_float)currentPhysicalQuantity.userRange.first;
  cl_float maxVal = (cl_float)currentPhysicalQuantity.userRange.second;
  clContext.setKernelArg(KERNEL_FILL_COLOR, 0, currentPhysicalQuantity.bufferName);
  clContext.setKernelArg(KERNEL_FILL_COLOR, 1, sizeof(cl_float), &minVal);
  clContext.setKernelArg(KERNEL_FILL_COLOR, 2, sizeof(cl_float), &maxVal);
  clContext.runKernel(KERNEL_FILL_COLOR, m_currNbParticles);

  // Rendering purpose
  clContext.runKernel(KERNEL_FILL_CAMERA_DIST, m_currNbParticles);

  m_radixSort.sort("p_cameraDist", { "p_pos", "p_col", "p_vel", "p_predPos" }, { "p_temp", "p_buoyancy", "p_vaporDens", "p_cloudDens", "p_partID" });

  clContext.releaseGLBuffers({ "p_pos", "p_col", "c_partDetector", "u_cameraPos" });
}

// Storing definitions here to prevent cl_types in headers
void Clouds::setRestDensity(float restDensity)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->restDensity = (cl_float)restDensity;
  updateFluidsParamsInKernels();
}

//
void Clouds::setRelaxCFM(float relaxCFM)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->relaxCFM = (cl_float)relaxCFM;
  updateFluidsParamsInKernels();
}

//
void Clouds::setTimeStep(float timeStep)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->timeStep = (cl_float)timeStep;
  updateFluidsParamsInKernels();
}

//
void Clouds::setNbJacobiIters(size_t nbIters)
{
  if (!m_init)
    return;
  m_nbJacobiIters = nbIters;
}

//
void Clouds::enableArtPressure(bool enable)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->isArtPressureEnabled = (cl_uint)enable;
  updateFluidsParamsInKernels();
}

//
void Clouds::setArtPressureRadius(float radius)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->artPressureRadius = (cl_float)radius;
  updateFluidsParamsInKernels();
}

//
void Clouds::setArtPressureExp(size_t exp)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->artPressureExp = (cl_uint)exp;
  updateFluidsParamsInKernels();
}

//
void Clouds::setArtPressureCoeff(float coeff)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->artPressureCoeff = (cl_float)coeff;
  updateFluidsParamsInKernels();
}

//
void Clouds::enableVorticityConfinement(bool enable)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->isVorticityConfEnabled = (cl_uint)enable;
  updateFluidsParamsInKernels();
}

//
void Clouds::setVorticityConfinementCoeff(float coeff)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->vorticityConfCoeff = (cl_float)coeff;
  updateFluidsParamsInKernels();
}

//
void Clouds::setXsphViscosityCoeff(float coeff)
{
  if (!m_init)
    return;
  m_fluidKernelInputs->xsphViscosityCoeff = (cl_float)coeff;
  updateFluidsParamsInKernels();
}

//
void Clouds::setGroundHeatCoeff(float coeff)
{
  if (!m_init)
    return;
  m_cloudKernelInputs->groundHeatCoeff = (cl_float)coeff;
  updateCloudsParamsInKernels();
}

//
void Clouds::setBuoyancyCoeff(float coeff)
{
  if (!m_init)
    return;
  m_cloudKernelInputs->buoyancyCoeff = (cl_float)coeff;
  updateCloudsParamsInKernels();
}

//
void Clouds::setAdiabaticLapseRate(float rate)
{
  if (!m_init)
    return;
  m_cloudKernelInputs->adiabaticLapseRate = (cl_float)rate;
  updateCloudsParamsInKernels();
}

//
void Clouds::setPhaseTransitionRate(float rate)
{
  if (!m_init)
    return;
  m_cloudKernelInputs->phaseTransitionRate = (cl_float)rate;
  updateCloudsParamsInKernels();
}

//
void Clouds::setLatentHeatCoeff(float coeff)
{
  if (!m_init)
    return;
  m_cloudKernelInputs->latentHeatCoeff = (cl_float)coeff;
  updateCloudsParamsInKernels();
}

//
float Clouds::getRestDensity() const
{
  return m_init ? (float)m_fluidKernelInputs->restDensity : 0.0f;
}

//
float Clouds::getRelaxCFM() const
{
  return m_init ? (float)m_fluidKernelInputs->relaxCFM : 0.0f;
}

//
float Clouds::getTimeStep() const
{
  return m_init ? (float)m_fluidKernelInputs->timeStep : 0.0f;
}

//
size_t Clouds::getNbJacobiIters() const
{
  return m_init ? m_nbJacobiIters : 0;
}

//
bool Clouds::isArtPressureEnabled() const
{
  return m_init ? (bool)m_fluidKernelInputs->isArtPressureEnabled : false;
}

//
float Clouds::getArtPressureRadius() const
{
  return m_init ? (float)m_fluidKernelInputs->artPressureRadius : 0.0f;
}

//
size_t Clouds::getArtPressureExp() const
{
  return m_init ? (size_t)m_fluidKernelInputs->artPressureExp : 0;
}

//
float Clouds::getArtPressureCoeff() const
{
  return m_init ? (float)m_fluidKernelInputs->artPressureCoeff : 0.0f;
}

//
bool Clouds::isVorticityConfinementEnabled() const
{
  return m_init ? (bool)m_fluidKernelInputs->isVorticityConfEnabled : 0.0f;
}

//
float Clouds::getVorticityConfinementCoeff() const
{
  return m_init ? (float)m_fluidKernelInputs->vorticityConfCoeff : 0.0f;
}

//
float Clouds::getXsphViscosityCoeff() const
{
  return m_init ? (float)m_fluidKernelInputs->xsphViscosityCoeff : 0.0f;
}

//
float Clouds::getGroundHeatCoeff() const
{
  return m_init ? (float)m_cloudKernelInputs->groundHeatCoeff : 0.0f;
}

//
float Clouds::getBuoyancyCoeff() const
{
  return m_init ? (float)m_cloudKernelInputs->buoyancyCoeff : 0.0f;
}

//
float Clouds::getAdiabaticLapseRate() const
{
  return m_init ? (float)m_cloudKernelInputs->adiabaticLapseRate : 0.0f;
}

//
float Clouds::getPhaseTransitionRate() const
{
  return m_init ? (float)m_cloudKernelInputs->phaseTransitionRate : 0.0f;
}

//
float Clouds::getLatentHeatCoeff() const
{
  return m_init ? (float)m_cloudKernelInputs->latentHeatCoeff : 0.0f;
}
