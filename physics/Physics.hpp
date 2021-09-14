#pragma once

#include "Math.hpp"
#include <array>

namespace Core
{
enum class Dimension
{
  dim2D,
  dim3D
};

enum class Boundary
{
  BouncingWall,
  CyclicWall
};

struct ModelParams
{
  size_t currNbParticles = 0;
  size_t maxNbParticles = 0;
  size_t boxSize = 0;
  size_t gridRes = 0;
  float velocity = 0.0f;
  unsigned int particleVBO = 0;
  unsigned int cameraVBO = 0;
  unsigned int gridVBO = 0;
};

class Physics
{
  public:
  Physics(ModelParams params, Dimension dimension = Dimension::dim2D)
      : m_maxNbParticles(params.maxNbParticles)
      , m_currNbParticles(params.currNbParticles)
      , m_boxSize(params.boxSize)
      , m_gridRes(params.gridRes)
      , m_nbCells(params.gridRes * params.gridRes * params.gridRes)
      , m_velocity(params.velocity)
      , m_particleVBO(params.particleVBO)
      , m_cameraVBO(params.cameraVBO)
      , m_gridVBO(params.gridVBO)
      , m_dimension(dimension)
      , m_boundary(Boundary::BouncingWall)
      , m_init(false)
      , m_pause(false) {};

  virtual ~Physics() = default;

  size_t maxNbParticles() const { return m_maxNbParticles; }

  void setNbParticles(size_t nbSelParticles) { m_currNbParticles = nbSelParticles; }
  size_t nbParticles() const { return m_currNbParticles; }

  void setDimension(Dimension dim)
  {
    m_dimension = dim;
    reset();
  }
  Dimension dimension() const { return m_dimension; }

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

  protected:
  bool m_init;
  bool m_pause;

  size_t m_maxNbParticles;
  size_t m_currNbParticles;

  size_t m_boxSize;

  size_t m_gridRes;
  size_t m_nbCells;

  float m_velocity;

  Dimension m_dimension;

  Boundary m_boundary;

  // Gate to graphics
  unsigned int m_particleVBO;
  unsigned int m_cameraVBO;
  unsigned int m_gridVBO;
};
}
