#include "Target.hpp"
#include "Logging.hpp"
#include <algorithm>
#include <cmath>
#include <math.h>

using namespace Physics;

Target::Target(size_t boxSize, Math::float3 initPos)
    : m_pNoiseTheta(PerlinNoise(1))
    , m_pNoiseBeta(PerlinNoise(29))
    , m_pNoiseR(PerlinNoise(246))
    , m_pos(initPos)
    , m_isActive(false)
    , m_isVisible(false)
    , m_radiusEffect(2.0f)
    , m_signEffect(1)
{
  m_maxRadius = 0.48f * boxSize;
  m_perlinPos = m_pos;
};

static constexpr float PI = 3.14f;

void Target::updatePos(Dimension dim, float particlesVel)
{
  // Generating perlin noise moving in a cartesian 3D grid with pseudo random gradients value at grid vertices
  float dx = 0.001f;
  m_perlinPos.x += dx;

  float dy = 0.002f;
  m_perlinPos.y += dy;

  float dz = 0.01f;
  m_perlinPos.z += dz;

  float nTheta = m_pNoiseTheta.computeNoiseValue(m_perlinPos);
  float nBeta = m_pNoiseBeta.computeNoiseValue(m_perlinPos);
  float nR = m_pNoiseR.computeNoiseValue(m_perlinPos);

  // Adjusting target velocity to match followers ones
  float velRatio = particlesVel / 35.0f;
  // Mapping it to spheric coordinates to generate a pseudo random 3D trajectory remaining inside radius
  float radius = m_maxRadius * cos(12 * velRatio * PI * nR);
  m_pos.x = (dim == Dimension::dim3D) ? (radius * cos(5 * velRatio * PI * nBeta)) : 0.0f;
  m_pos.y = radius * sin(5 * velRatio * PI * nBeta) * cos(4 * velRatio * PI * nTheta);
  m_pos.z = radius * sin(5 * velRatio * PI * nBeta) * sin(4 * velRatio * PI * nTheta);
}