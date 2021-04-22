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
  PerlinParticle();
  ~PerlinParticle() = default;

  void update();

  private:
  Math::float3 m_pos;
};
}