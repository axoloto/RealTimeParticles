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
#define KERNEL_PREDICT_POS "predictPosition"
#define KERNEL_APPLY_BOUNDARY "applyBoundaryCondition"
#define KERNEL_DENSITY "computeDensity"
#define KERNEL_CONSTRAINT_FACTOR "computeConstraintFactor"
#define KERNEL_CONSTRAINT_CORRECTION "computeConstraintCorrection"
#define KERNEL_CORRECT_POS "correctPosition"
#define KERNEL_UPDATE_VEL "updateVel"
#define KERNEL_COMPUTE_VORTICITY "computeVorticity"
#define KERNEL_VORTICITY_CONFINEMENT "applyVorticityConfinement"
#define KERNEL_XSPH_VISCOSITY "applyXsphViscosityCorrection"
#define KERNEL_UPDATE_POS "updatePosition"
#define KERNEL_FILL_COLOR "fillFluidColor"

namespace Physics
{
struct FluidKernelInputs
{
  cl_float effectRadius = 0.3f;
  cl_float restDensity = 450.0f;
  cl_float relaxCFM = 600.0f;
  cl_float timeStep = 0.010f;
  cl_uint dim = 3;
  // Artifical pressure if enabled will try to reduce tensile instability
  cl_uint isArtPressureEnabled = 1;
  cl_float artPressureRadius = 0.006f;
  cl_float artPressureCoeff = 0.001f;
  cl_uint artPressureExp = 4;
  // Vorticity confinement if enabled will try to replace lost energy due to virtual damping
  cl_uint isVorticityConfEnabled = 1;
  cl_float vorticityConfCoeff = 0.0004f;
  cl_float xsphViscosityCoeff = 0.0001f;
};

const std::map<Clouds::CaseType, std::string, Clouds::CompareCaseType> Clouds::ALL_CASES {
  { CaseType::CUMULUS, "Cumulus" }
};
}

Clouds::Clouds(ModelParams params)
    : Model(params)
    , m_simplifiedMode(true)
    , m_maxNbPartsInCell(100)
    , m_radixSort(params.maxNbParticles)
    , m_kernelInputs(std::make_unique<FluidKernelInputs>())
    , m_initialCase(CaseType::CUMULUS)
    , m_nbJacobiIters(2)
{
  createProgram();

  createBuffers();

  createKernels();

  m_init = (m_kernelInputs != nullptr);

  reset();
}

// Must be on implementation side as FluidKernelInputs must be complete
Clouds::~Clouds() {};

bool Clouds::createProgram() const
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
  clBuildOptions << " -DPOLY6_COEFF=" << Utils::FloatToStr(315.0f / (64.0f * Math::PI_F * std::pow(effectRadius, 9.f)));
  clBuildOptions << " -DSPIKY_COEFF=" << Utils::FloatToStr(15.0f / (Math::PI_F * std::pow(effectRadius, 6.f)));
  clBuildOptions << " -DMAX_VEL=" << Utils::FloatToStr(30.0f);

  LOG_INFO(clBuildOptions.str());
  clContext.createProgram(PROGRAM_CLOUDS, std::vector<std::string>({ "fluids.cl", "clouds.cl", "utils.cl", "grid.cl" }), clBuildOptions.str());

  return true;
}

bool Clouds::createBuffers() const
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
  clContext.createBuffer("p_velInViscosity", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_vort", 4 * m_maxNbParticles * sizeof(float), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cellID", m_maxNbParticles * sizeof(unsigned int), CL_MEM_READ_WRITE);
  clContext.createBuffer("p_cameraDist", m_maxNbParticles * sizeof(unsigned int), CL_MEM_READ_WRITE);

  clContext.createBuffer("c_startEndPartID", 2 * m_nbCells * sizeof(unsigned int), CL_MEM_READ_WRITE);

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
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_COLOR, { "p_density", "", "p_col" });

  // Radix Sort based on 3D grid, using predicted positions, not corrected ones
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_RESET_CELL_ID, { "p_cellID" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_CELL_ID, { "p_predPos", "p_cellID" });

  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_RESET_START_END_CELL, { "c_startEndPartID" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_START_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_FILL_END_CELL, { "p_cellID", "c_startEndPartID" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_ADJUST_END_CELL, { "c_startEndPartID" });

  // Position Based Fluids
  /// Position prediction
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_PREDICT_POS, { "p_pos", "p_vel", "", "p_predPos" });
  /// Boundary conditions
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_APPLY_BOUNDARY, { "p_predPos" });
  /// Jacobi solver to correct position
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_DENSITY, { "p_predPos", "c_startEndPartID", "", "p_density" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CONSTRAINT_FACTOR, { "p_predPos", "p_density", "c_startEndPartID", "", "p_constFactor" });
  clContext.createKernel(PROGRAM_CLOUDS, KERNEL_CONSTRAINT_CORRECTION, { "p_constFactor", "c_startEndPartID", "p_predPos", "", "p_corrPos" });
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

void Clouds::updateCloudsParamsInKernel()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  m_kernelInputs->dim = (m_dimension == Dimension::dim2D) ? 2 : 3;

  const float effectRadius = ((float)m_boxSize) / m_gridRes;
  m_kernelInputs->effectRadius = effectRadius;

  clContext.setKernelArg(KERNEL_RANDOM_POS, 0, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_PREDICT_POS, 2, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_UPDATE_VEL, 2, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_DENSITY, 2, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_CONSTRAINT_FACTOR, 3, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_CONSTRAINT_CORRECTION, 3, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_FILL_COLOR, 1, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_COMPUTE_VORTICITY, 3, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_VORTICITY_CONFINEMENT, 3, sizeof(FluidKernelInputs), m_kernelInputs.get());
  clContext.setKernelArg(KERNEL_XSPH_VISCOSITY, 3, sizeof(FluidKernelInputs), m_kernelInputs.get());
}

void Clouds::reset()
{
  if (!m_init)
    return;

  CL::Context& clContext = CL::Context::Get();

  updateCloudsParamsInKernel();

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

  if (m_dimension == Dimension::dim2D)
  {
    Geometry::Shape2D shape = Geometry::Shape2D::Rectangle;

    switch (m_initialCase)
    {
    case CaseType::CUMULUS:
      m_currNbParticles = Utils::NbParticles::P4K;

      shape = Geometry::Shape2D::Rectangle;
      startFluidPos = { 0.0f, m_boxSize / -2.0f, m_boxSize / -2.0f };
      endFluidPos = { 0.0f, 0.0f, 0.0f };
      break;
    default:
      LOG_ERROR("Unkown case type");
      break;
    }

    const auto& subdiv2D = Utils::GetNbParticlesSubdiv2D((Utils::NbParticles)m_currNbParticles);
    Math::int2 grid2DRes = { subdiv2D[0], subdiv2D[1] };

    gridVerts = Geometry::Generate2DGrid(shape, Geometry::Plane::YZ, grid2DRes, startFluidPos, endFluidPos);
  }
  else if (m_dimension == Dimension::dim3D)
  {
    Geometry::Shape3D shape = Geometry::Shape3D::Box;

    switch (m_initialCase)
    {
    case CaseType::CUMULUS:
      m_currNbParticles = Utils::NbParticles::P130K;
      shape = Geometry::Shape3D::Box;
      startFluidPos = { m_boxSize / -2.0f, m_boxSize / -2.0f, m_boxSize / -2.0f };
      endFluidPos = { m_boxSize / 2.0f, 0.0f, 0.0f };
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
      [](const Math::float3& vertPos) -> std::array<float, 4>
      { return { vertPos.x, vertPos.y, vertPos.z, 0.0f }; });

  clContext.loadBufferFromHost("p_pos", 0, 4 * sizeof(float) * pos.size(), pos.data());

  std::vector<std::array<float, 4>> vel(m_maxNbParticles, std::array<float, 4>({ 0.0f, 0.0f, 0.0f, 0.0f }));
  clContext.loadBufferFromHost("p_vel", 0, 4 * sizeof(float) * vel.size(), vel.data());

  std::vector<std::array<float, 4>> col(m_maxNbParticles, std::array<float, 4>({ 0.0f, 0.1f, 1.0f, 0.0f }));
  clContext.loadBufferFromHost("p_col", 0, 4 * sizeof(float) * col.size(), col.data());

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

    // Updating velocity
    clContext.runKernel(KERNEL_UPDATE_VEL, m_currNbParticles);

    if (m_kernelInputs->isVorticityConfEnabled)
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
    clContext.runKernel(KERNEL_FILL_COLOR, m_currNbParticles);
  }

  // Rendering purpose
  clContext.runKernel(KERNEL_FILL_CAMERA_DIST, m_currNbParticles);

  m_radixSort.sort("p_cameraDist", { "p_pos", "p_col", "p_vel", "p_predPos" });

  clContext.releaseGLBuffers({ "p_pos", "p_col", "c_partDetector", "u_cameraPos" });
}

// Storing definitions here to prevent cl_types in headers
void Clouds::setRestDensity(float restDensity)
{
  if (!m_init)
    return;
  m_kernelInputs->restDensity = (cl_float)restDensity;
  updateCloudsParamsInKernel();
}

//
void Clouds::setRelaxCFM(float relaxCFM)
{
  if (!m_init)
    return;
  m_kernelInputs->relaxCFM = (cl_float)relaxCFM;
  updateCloudsParamsInKernel();
}

//
void Clouds::setTimeStep(float timeStep)
{
  if (!m_init)
    return;
  m_kernelInputs->timeStep = (cl_float)timeStep;
  updateCloudsParamsInKernel();
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
  m_kernelInputs->isArtPressureEnabled = (cl_uint)enable;
  updateCloudsParamsInKernel();
}

//
void Clouds::setArtPressureRadius(float radius)
{
  if (!m_init)
    return;
  m_kernelInputs->artPressureRadius = (cl_float)radius;
  updateCloudsParamsInKernel();
}

//
void Clouds::setArtPressureExp(size_t exp)
{
  if (!m_init)
    return;
  m_kernelInputs->artPressureExp = (cl_uint)exp;
  updateCloudsParamsInKernel();
}

//
void Clouds::setArtPressureCoeff(float coeff)
{
  if (!m_init)
    return;
  m_kernelInputs->artPressureCoeff = (cl_float)coeff;
  updateCloudsParamsInKernel();
}

//
void Clouds::enableVorticityConfinement(bool enable)
{
  if (!m_init)
    return;
  m_kernelInputs->isVorticityConfEnabled = (cl_uint)enable;
  updateCloudsParamsInKernel();
}

//
void Clouds::setVorticityConfinementCoeff(float coeff)
{
  if (!m_init)
    return;
  m_kernelInputs->vorticityConfCoeff = (cl_float)coeff;
  updateCloudsParamsInKernel();
}

//
void Clouds::setXsphViscosityCoeff(float coeff)
{
  if (!m_init)
    return;
  m_kernelInputs->xsphViscosityCoeff = (cl_float)coeff;
  updateCloudsParamsInKernel();
}

// Not giving access to it for now.
// Strongly connected to grid resolution which is not available as parameter,
// in order to maintain cohesion between boids and clouds models
/*
void Clouds::setEffectRadius(float effectRadius)
{
  if(!m_init) return;
  m_kernelInputs.effectRadius = (cl_float)effectRadius;
  updateCloudsParamsInKernel();
}
*/
float Clouds::getEffectRadius() const { return m_init ? (float)m_kernelInputs->effectRadius : 0.0f; }

//
float Clouds::getRestDensity() const { return m_init ? (float)m_kernelInputs->restDensity : 0.0f; }

//
float Clouds::getRelaxCFM() const { return m_init ? (float)m_kernelInputs->relaxCFM : 0.0f; }

//
float Clouds::getTimeStep() const { return m_init ? (float)m_kernelInputs->timeStep : 0.0f; }

//
size_t Clouds::getNbJacobiIters() const { return m_init ? m_nbJacobiIters : 0; }

//
bool Clouds::isArtPressureEnabled() const { return m_init ? (bool)m_kernelInputs->isArtPressureEnabled : false; }

//
float Clouds::getArtPressureRadius() const { return m_init ? (float)m_kernelInputs->artPressureRadius : 0.0f; }

//
size_t Clouds::getArtPressureExp() const { return m_init ? (size_t)m_kernelInputs->artPressureExp : 0; }

//
float Clouds::getArtPressureCoeff() const { return m_init ? (float)m_kernelInputs->artPressureCoeff : 0.0f; }

//
bool Clouds::isVorticityConfinementEnabled() const { return m_init ? (bool)m_kernelInputs->isVorticityConfEnabled : 0.0f; }

//
float Clouds::getVorticityConfinementCoeff() const { return m_init ? (float)m_kernelInputs->vorticityConfCoeff : 0.0f; }

//
float Clouds::getXsphViscosityCoeff() const { return m_init ? (float)m_kernelInputs->xsphViscosityCoeff : 0.0f; }
