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

class Physics
{
  public:
  Physics(size_t maxNbParticles, size_t nbParticles, size_t boxSize, size_t gridRes, float velocity, Dimension dimension = Dimension::dim2D)
      : m_maxNbParticles(maxNbParticles)
      , m_nbParticles(nbParticles)
      , m_boxSize(boxSize)
      , m_gridRes(gridRes)
      , m_nbCells(gridRes * gridRes * gridRes)
      , m_init(false)
      , m_velocity(velocity)
      , m_dimension(dimension)
      , m_boundary(Boundary::BouncingWall)
      , m_pause(false)
      , m_activeTarget(true) {};

  virtual ~Physics() = default;

  size_t maxNbParticles() const { return m_maxNbParticles; }

  void setNbParticles(size_t nbSelParticles) { m_nbParticles = nbSelParticles; }
  size_t nbParticles() const { return m_nbParticles; }

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

  virtual void activateTarget(bool target) { m_activeTarget = target; }
  bool isTargetActivated() const { return m_activeTarget; }

  virtual Math::float3 targetPos() const { return { 0.0f, 0.0f, 0.0f }; }

  protected:
  bool m_init;
  bool m_pause;

  size_t m_maxNbParticles;
  size_t m_nbParticles;

  size_t m_boxSize;

  size_t m_gridRes;
  size_t m_nbCells;

  float m_velocity;

  Dimension m_dimension;

  Boundary m_boundary;

  bool m_activeTarget;
};
}
