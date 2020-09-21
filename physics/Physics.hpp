#pragma once

#include "diligentGraphics/Math.hpp"
#include <array>

namespace Core
{
static constexpr int NUM_MAX_ENTITIES = 100000;

enum Dimension
{
  dim2D,
  dim3D
};

class Physics
{
  public:
  Physics(int boxSize, int numEntities, Dimension dimension = Dimension::dim3D);
  ~Physics() = default;

  void* getCoordsBufferStart();
  void* getColorsBufferStart();

  int numMaxEntities() const { return NUM_MAX_ENTITIES; }

  int numEntities() const { return m_numEntities; }
  void setNumEntities(int numEntities) { m_numEntities = numEntities; }

  virtual void updatePhysics() = 0;

  void update();
  void updateBuffers();

  void resetParticles();

  void setDimension(Dimension dim)
  {
    m_dimension = dim;
    resetParticles();
  }
  Dimension getDimension() const { return m_dimension; }

  void setPause(bool pause) { m_pause = pause; }
  float getPause() { return m_pause; }

  void forceMaxSpeed(bool forcedmax) { m_forceMaxSpeed = forcedmax; }
  float isMaxSpeedForced() { return m_forceMaxSpeed; }

  void setMaxVelocity(float maxVelocity) { m_maxSpeed = maxVelocity; }
  float getmaxVelocity() { return m_maxSpeed; }

  void setBouncingWall(bool bouncingwall) { m_activateBouncingWall = bouncingwall; }
  float getBouncingWall() { return m_activateBouncingWall; }

  void setCyclicWall(bool Cyclicwall) { m_activateCyclicWall = Cyclicwall; }
  float getCyclicWall() { return m_activateCyclicWall; }

  protected:
  struct Entity
  {
    // Mandatory
    Math::float3 xyz;
    // Mandatory
    Math::float3 rgb;

    Math::float3 vxyz;
    Math::float3 axyz;
  };

  std::array<Entity, NUM_MAX_ENTITIES> m_entities;

  std::array<std::array<float, 4>, NUM_MAX_ENTITIES> m_coordsBuffer;
  std::array<std::array<float, 3>, NUM_MAX_ENTITIES> m_colorsBuffer;

  int m_numEntities;
  int m_boxSize;

  void updateParticle(Entity& particle);
  void bouncingWall(Entity& particle);
  void cyclicWall(Entity& particle);
  void randomWall(Entity& particle); // WIP

  float m_maxSpeed;

  Dimension m_dimension;
  bool m_activateBouncingWall;
  bool m_activateCyclicWall;
  bool m_forceMaxSpeed;
  bool m_pause;
};
}
