#pragma once

#include <array>
#include <vector>

#include "Math.hpp"
#include "PerlinNoise.hpp"

namespace Core
{
class Target
{
  public:
  Target(size_t boxSize, Math::float3 initPos = { 0.0f, 0.0f, 0.0f });
  ~Target() = default;

  Math::float3 pos() const { return m_pos; }
  void updatePos(float velocity);

  void activate(bool isActive) { m_isActive = isActive; }
  bool isActivated() const { return m_isActive; }

  void setRadiusEffect(float radiusEffect) { m_radiusEffect = radiusEffect; }
  float radiusEffect() const { return m_radiusEffect; }

  void setSignEffect(int signEffect) { m_signEffect = signEffect; }
  int signEffect() const { return m_signEffect; }

  private:
  Math::float3 m_perlinPos;
  PerlinNoise m_pNoiseTheta;
  PerlinNoise m_pNoiseBeta;
  PerlinNoise m_pNoiseR;

  float m_maxRadius;
  Math::float3 m_pos;

  bool m_isActive;
  float m_radiusEffect;
  int m_signEffect;
};
}