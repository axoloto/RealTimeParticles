#pragma once

#include "diligentGraphics/Math.hpp"
#include <array>

namespace Core
{
static constexpr int NUM_MAX_ENTITIES = 30000;

enum Dimension
{
  dim2D,
  dim3D
};

class Physics
{
  public:
  Physics(int numEntities, Dimension dimension = Dimension::dim2D)
      : m_numEntities(numEntities)
      , m_velocity(4.0f)
      , m_dimension(dimension)
      , m_activateBouncingWall(false)
      , m_activateCyclicWall(true)
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

  virtual void update() = 0;
  virtual void reset() = 0;

  void pause(bool pause) { m_pause = pause; }
  bool onPause() const { return m_pause; }

  void setVelocity(float velocity) { m_velocity = velocity; }
  float velocity() { return m_velocity; }

  void setBouncingWall(bool bouncingwall) { m_activateBouncingWall = bouncingwall; }
  bool isBouncingWallEnabled() const { return m_activateBouncingWall; }

  void setCyclicWall(bool Cyclicwall) { m_activateCyclicWall = Cyclicwall; }
  bool isCyclicWallEnabled() const { return m_activateCyclicWall; }

  protected:
  int m_numEntities;
  float m_velocity;
  Dimension m_dimension;
  bool m_activateBouncingWall;
  bool m_activateCyclicWall;
  bool m_pause;
};
}
