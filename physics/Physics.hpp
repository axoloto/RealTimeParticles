#pragma once

#include "Math.hpp"
#include <array>

namespace Core
{
static constexpr int NUM_MAX_ENTITIES = 30000;

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
  Physics(int numEntities, Dimension dimension = Dimension::dim2D)
      : m_numEntities(numEntities)
      , m_init(false)
      , m_velocity(3.0f)
      , m_dimension(dimension)
      , m_boundary(Boundary::BouncingWall)
      , m_pause(false) {};

  virtual ~Physics() = default;

  void setNumEntities(int numEntities) { m_numEntities = numEntities; }
  int numEntities() const { return m_numEntities; }

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

  protected:
  bool m_init;
  int m_numEntities;
  float m_velocity;
  Dimension m_dimension;
  Boundary m_boundary;
  bool m_pause;
};
}
