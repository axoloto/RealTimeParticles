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
  Physics(Dimension dimension = Dimension::dim2D)
      : m_velocity(4.0f)
      , m_dimension(dimension)
      , m_activateBouncingWall(false)
      , m_activateCyclicWall(true)
      , m_pause(false) {};
  virtual ~Physics() = default;

  void setDimension(Dimension dim)
  {
    m_dimension = dim;
    reset();
  }
  Dimension getDimension() const { return m_dimension; }

  virtual void update() = 0;
  virtual void reset() = 0;

  void setPause(bool pause) { m_pause = pause; }
  float getPause() { return m_pause; }

  void setVelocity(float velocity) { m_velocity = velocity; }
  float getVelocity() { return m_velocity; }

  void setBouncingWall(bool bouncingwall) { m_activateBouncingWall = bouncingwall; }
  float getBouncingWall() { return m_activateBouncingWall; }

  void setCyclicWall(bool Cyclicwall) { m_activateCyclicWall = Cyclicwall; }
  float getCyclicWall() { return m_activateCyclicWall; }

  protected:
  float m_velocity;
  Dimension m_dimension;
  bool m_activateBouncingWall;
  bool m_activateCyclicWall;
  bool m_pause;
};
}
