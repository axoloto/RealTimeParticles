#include "PerlinParticle.hpp"
#include <algorithm>
#include <cmath>
#include <math.h>
#include <spdlog/spdlog.h>

using namespace Core;

PerlinParticle::PerlinParticle(size_t boxSize, Math::float3 initPos)
    : m_pNoiseX(PerlinNoise(1))
    , m_pNoiseY(PerlinNoise(29))
    , m_pNoiseZ(PerlinNoise(246))
    , m_pos(initPos)
{
  m_perlinPos = m_pos;
  //m_radius
};

static constexpr float PI = 3.14;

void PerlinParticle::updatePos(float velocity)
{
  // Tweaking perlin noise to generate a pseudo random 3D trajectory remaining inside given box
  float dx = 0.001f;
  float dy = 0.002f;
  float dz = 0.01f;

  m_perlinPos.x += dx;
  m_perlinPos.y += dy;
  m_perlinPos.z += dz;
  float nx = m_pNoiseX.computeNoiseValue(m_perlinPos);
  float ny = m_pNoiseY.computeNoiseValue(m_perlinPos);

  float radius = 600 * cos(6 * PI * nx);
  m_pos.x = radius * cos(5 * PI * ny);
  m_pos.y = radius * sin(5 * PI * ny) * cos(4 * PI * nx);
  m_pos.z = radius * sin(5 * PI * ny) * sin(4 * PI * nx);
}