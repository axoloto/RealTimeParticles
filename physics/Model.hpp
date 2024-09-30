#pragma once

#include "Geometry.hpp"
#include "Math.hpp"
#include "Parameters.hpp"
//#include <nlohmann/json.hpp>
//using json = nlohmann::ordered_json;

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace Physics
{
// List of supported physical models
enum ModelType
{
  BOIDS = 0,
  FLUIDS = 1,
  CLOUDS = 2
};

struct CompareModelType
{
  bool operator()(const ModelType& modelA, const ModelType& modelB) const
  {
    return (int)modelA < (int)modelB;
  }
};

static const std::map<ModelType, std::string, CompareModelType> ALL_MODELS {
  { ModelType::BOIDS, "Boids" }, // Craig Reynolds boids laws, 1987
  { ModelType::FLUIDS, "Fluids" }, // Position Based Fluids by NVIDIA team (Macklin and Muller, 2013)
  { ModelType::CLOUDS, "Clouds" }, // Position Based Fluids + Clouds Physics + Constrained (smoothed) temperature field (CWT Barbosa, Dobashi & Yamamoto, 2015)
};

// Boundary Condition types
enum class Boundary
{
  // Particle bounces the wall and goes into the other direction
  BouncingWall,
  // Particle goes through the wall to appear at the opposite wall
  CyclicWall
};

// Quantified physical properties that can be visualized at UI level through particles color
struct PhysicalQuantity
{
  // Name of the physical quantity (visible in UI)
  const std::string name;
  // Name of the OpenCL buffer containing the values of this quantity for all the particles
  const std::string bufferName;
  // For rendering purpose, color intensity will vary between two user-selected values (umin/umax)
  // Those user-selected values will vary inside the developer-defined static range (smin/smax) defined below
  std::pair<float, float> staticRange;
  // The user-selected values are defined here, umin belongs to [smin, umax] | umax belongs to [umin, smax]
  std::pair<float, float> userRange;
};

struct ModelParams
{
  size_t currNbParticles = 0;
  size_t maxNbParticles = 0;
  Geometry::BoxSize3D boxSize = { 0, 0, 0 };
  Geometry::BoxSize3D gridRes = { 0, 0, 0 };
  float velocity = 0.0f;
  unsigned int particlePosVBO = 0;
  unsigned int particleColVBO = 0;
  unsigned int cameraVBO = 0;
  unsigned int gridVBO = 0;
  Geometry::Dimension dimension = Geometry::Dimension::dim3D;
  Utils::PhysicsCase pCase = Utils::PhysicsCase::CASE_INVALID;
};

// Models Factory
class Model;
std::unique_ptr<Model> CreateModel(ModelType type, ModelParams params);

// Abstract class defining physical model foundations to implement
// Currently all models are OpenCL-based but that could change
class Model
{
  public:
  Model(ModelParams params, json js = {})
      : m_maxNbParticles(params.maxNbParticles)
      , m_currNbParticles(params.currNbParticles)
      , m_boxSize(params.boxSize)
      , m_gridRes(params.gridRes)
      , m_nbCells(params.gridRes.x * params.gridRes.y * params.gridRes.z)
      , m_velocity(params.velocity)
      , m_particlePosVBO(params.particlePosVBO)
      , m_particleColVBO(params.particleColVBO)
      , m_cameraVBO(params.cameraVBO)
      , m_gridVBO(params.gridVBO)
      , m_dimension(params.dimension)
      , m_case(params.pCase)
      , m_boundary(Boundary::BouncingWall)
      , m_init(false)
      , m_pause(false)
      , m_currentDisplayedQuantityName("")
      , m_inputJson(js) {};

  virtual ~Model() {};

  size_t maxNbParticles() const { return m_maxNbParticles; }

  void setNbParticles(size_t nbSelParticles) { m_currNbParticles = nbSelParticles; }
  size_t nbParticles() const { return m_currNbParticles; }

  void setDimension(Geometry::Dimension dim)
  {
    m_dimension = dim;
    reset();
  }
  Geometry::Dimension dimension() const { return m_dimension; }

  void setBoundary(Boundary boundary)
  {
    m_boundary = boundary;
  }
  Boundary boundary() const { return m_boundary; }

  virtual void update() = 0;
  virtual void reset() = 0;

  bool isInit() const { return m_init; }

  void pause(bool pause) { m_pause = pause; }
  bool onPause() const { return m_pause; }

  virtual void setVelocity(float velocity) { m_velocity = velocity; }
  float velocity() const { return m_velocity; }

  virtual Math::float3 targetPos() const { return { 0.0f, 0.0f, 0.0f }; }
  virtual bool isTargetActivated() const { return false; }
  virtual bool isTargetVisible() const { return false; }

  void setCurrentDisplayedQuantity(const std::string& name);
  // Name of currently displayed physical quantity, only one for which some specific fields can be modified
  std::string currentDisplayedPhysicalQuantityName() { return m_currentDisplayedQuantityName; }
  // Currently displayed physical quantity
  PhysicalQuantity& currentDisplayedPhysicalQuantity()
  {
    auto it = m_allDisplayableQuantities.find(m_currentDisplayedQuantityName);
    return (it != m_allDisplayableQuantities.end()) ? it->second : m_allDisplayableQuantities.begin()->second;
  }
  // All available physical quantities to be displayed, read-only
  std::map<const std::string, PhysicalQuantity>::iterator beginDisplayablePhysicalQuantities() { return m_allDisplayableQuantities.begin(); }
  std::map<const std::string, PhysicalQuantity>::iterator endDisplayablePhysicalQuantities() { return m_allDisplayableQuantities.end(); }

  std::map<const std::string, PhysicalQuantity>::const_iterator cbeginDisplayablePhysicalQuantities() { return m_allDisplayableQuantities.cbegin(); }
  std::map<const std::string, PhysicalQuantity>::const_iterator cendDisplayablePhysicalQuantities() { return m_allDisplayableQuantities.cend(); }

  virtual bool isProfilingEnabled() const { return false; };
  virtual void enableProfiling(bool enable) {};
  virtual bool isUsingIGPU() const { return false; };

  json getInputJson() const
  {
    return m_inputJson;
  }

  void updateInputJson(const json& newJson)
  {
    // No modification
    if (json::diff(m_inputJson, newJson).empty())
      return;

    m_inputJson.merge_patch(newJson);
    updateModelWithInputJson();
  }

  virtual void updateModelWithInputJson() {}; // = 0;

  void setCase(Utils::PhysicsCase caseType) { m_case = caseType; }
  const Utils::PhysicsCase getCase() const { return m_case; }

  protected:
  bool m_init;
  bool m_pause;

  size_t m_maxNbParticles;
  size_t m_currNbParticles;

  Geometry::BoxSize3D m_boxSize;

  Geometry::BoxSize3D m_gridRes;
  size_t m_nbCells;

  float m_velocity;

  Geometry::Dimension m_dimension;

  Utils::PhysicsCase m_case;

  Boundary m_boundary;

  // Gate to graphics
  unsigned int m_particlePosVBO;
  unsigned int m_particleColVBO;
  unsigned int m_cameraVBO;
  unsigned int m_gridVBO;

  // Name of the PhysicalQuantity currently sent to color buffer and rendered by fragment shader
  std::string m_currentDisplayedQuantityName;
  // All PhysicalQuantities that can be rendered
  std::map<const std::string, PhysicalQuantity> m_allDisplayableQuantities;

  // Container for model parameters available in UI
  json m_inputJson;
};
}
