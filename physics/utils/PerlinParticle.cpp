#include "PerlinParticle.hpp"
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

using namespace Core;

PerlinParticle::PerlinParticle(Math::float3 initPos)
    : m_pNoiseX(PerlinNoise(1))
    , m_pNoiseY(PerlinNoise(29))
    , m_pNoiseZ(PerlinNoise(246))
    , m_pos(initPos)
{
  m_perlinPos = m_pos;
};

void PerlinParticle::updatePos(float timeStep)
{
  float f = 0.01f;
  // Math::float3 perlinPos = { m_pos.x * f, m_pos.y * f, m_pos.z * f };
  spdlog::info("x  {},  y  {}, z  {}", m_pos.x, m_pos.y, m_pos.z);

  m_perlinPos.x += 0.001f;
  m_perlinPos.y += 0.001f;
  m_perlinPos.z += f;
  float nx = m_pNoiseX.computeNoiseValue(m_perlinPos);

  /*
  float nx = m_pNoiseX.computeNoiseValue(perlinPos);
  float ny = m_pNoiseY.computeNoiseValue(perlinPos);
  float nz = m_pNoiseZ.computeNoiseValue(perlinPos);
  spdlog::info("noise x {}, noise y {}, noise z {}", nx, ny, nz);

  if (std::abs(nx - 0.5) > 0.01f && std::abs(ny - 0.5) > 0.01f && std::abs(nz - 0.5f) > 0.01f)
  {
    m_vel.x = 6 * cos(10 * nx * 3.14);
    m_vel.y = 6 * sin(10 * ny * 2 * 3.14);
    m_vel.z = 10 * (nz - 0.5) * 2;
  }
*/

  float r = 600 * cos(6 * 3.14 * nx);
  m_pos.x = 0.0f;
  m_pos.y = r * cos(4 * 3.14 * nx);
  m_pos.z = r * sin(4 * 3.14 * nx);
  timeStep = 0.01;

  spdlog::info("vx  {},  vy  {}, vz  {}", m_vel.x, m_vel.y, m_vel.z);

  // m_pos.x += timeStep * m_vel.x;
  // m_pos.y += timeStep * m_vel.y;
  // m_pos.z += timeStep * m_vel.z;
}