#include "PerlinNoise.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

using namespace Physics;

PerlinNoise::PerlinNoise(int seed)
{
  m_perms.resize(256);

  // Fill p with values from 0 to 255
  std::iota(m_perms.begin(), m_perms.end(), 0);

  // Initialize a random engine with seed
  std::default_random_engine engine(seed);

  // Suffle  using the above random engine
  std::shuffle(m_perms.begin(), m_perms.end(), engine);

  // Duplicate the permutation vector
  m_perms.insert(m_perms.end(), m_perms.begin(), m_perms.end());
};

float PerlinNoise::computeNoiseValue(Math::float3 pos)
{
  float x = pos.x;
  float y = pos.y;
  float z = pos.z;

  // Find the unit cube that contains the point
  int X = (int)floor(x) & 255;
  int Y = (int)floor(y) & 255;
  int Z = (int)floor(z) & 255;

  // Find relative x, y, z of point in cube
  x -= floor(x);
  y -= floor(y);
  z -= floor(z);

  // Compute fade curves for each of x, y, z
  float u = fade(x);
  float v = fade(y);
  float w = fade(z);

  // Hash coordinates of the 8 cube corners
  int A = m_perms[X] + Y;
  int AA = m_perms[A] + Z;
  int AB = m_perms[A + 1] + Z;
  int B = m_perms[X + 1] + Y;
  int BA = m_perms[B] + Z;
  int BB = m_perms[B + 1] + Z;

  // Add blended results from 8 corners of cube
  float res = lerp(w,
      lerp(v, lerp(u, grad(m_perms[AA], x, y, z), grad(m_perms[BA], x - 1, y, z)),
          lerp(u, grad(m_perms[AB], x, y - 1, z), grad(m_perms[BB], x - 1, y - 1, z))),
      lerp(v, lerp(u, grad(m_perms[AA + 1], x, y, z - 1), grad(m_perms[BA + 1], x - 1, y, z - 1)),
          lerp(u, grad(m_perms[AB + 1], x, y - 1, z - 1), grad(m_perms[BB + 1], x - 1, y - 1, z - 1))));

  return (res + 1.0f) / 2.0f;
}

float PerlinNoise::fade(float t)
{
  return t * t * t * (t * (t * 6 - 15) + 10);
}

float PerlinNoise::lerp(float t, float a, float b)
{
  return a + t * (b - a);
}

float PerlinNoise::grad(int hash, float x, float y, float z)
{
  int h = hash & 15;

  // Convert lower 4 bits of hash into 12 gradient directions
  float u = h < 8 ? x : y,
        v = h < 4 ? y : h == 12 || h == 14 ? x
                                           : z;

  return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}