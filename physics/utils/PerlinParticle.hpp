#pragma once

#include <array>
#include <vector>

#include "Math.hpp"
#include "PerlinNoise.hpp"

namespace Core
{
class PerlinParticle
{
  public:
  PerlinParticle(size_t boxSize, Math::float3 initPos = { 100.0f, 0.0f, 0.0f });
  ~PerlinParticle() = default;

  Math::float3 pos() const { return m_pos; }
  void updatePos(float timeStep);

  private:
  Math::float3 m_pos;
  Math::float3 m_perlinPos;
  Math::float3 m_vel;

  PerlinNoise m_pNoiseX;
  PerlinNoise m_pNoiseY;
  PerlinNoise m_pNoiseZ;
};
}